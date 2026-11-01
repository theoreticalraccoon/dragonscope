"""FastAPI and WebSocket server around one booth session. See docs/PROTOCOL.md.

Run: ``python -m dragonscope.server``

It binds to 127.0.0.1 and keeps everything in memory. Access logs are off, and
nothing here logs message contents, so chat values do not reach the console.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from dragonscope.activity import model_facts
from dragonscope.session import DemoSession, SessionError, guide
from dragonscope.train import load_model
from dragonscope.vendor.bdh.bdh import BDH

logger = logging.getLogger("dragonscope.server")
MAX_MESSAGE_BYTES = 4096
STATIC_DIR = Path(__file__).parent / "static"
# Fastest in a test on the demo laptop: 17 ms a byte, against 23 ms at the default
# of 10 threads and 25 ms at 1.
CPU_THREADS = 4


def _same_origin(websocket: WebSocket) -> bool:
    """Stop other web pages in the browser from connecting to the local session."""
    origin = websocket.headers.get("origin")
    return origin is None or urlparse(origin).netloc == websocket.headers.get("host")


def _dispatch(session: DemoSession, message: dict) -> dict | None:
    """Apply one client message. Returns a ``forget_result`` when there is one."""
    kind = message.get("type")
    if kind == "say":
        value = message.get("value")
        if not isinstance(value, str):
            raise SessionError("invalid_value", "A value is required.")
        session.say(message.get("field"), value)
    elif kind == "ask":
        session.ask(message.get("field"))
    elif kind == "forget":
        result = session.forget(message.get("finding"))
        return {"type": "forget_result", **result}
    elif kind == "reset":
        session.reset()
    else:
        raise SessionError("unknown_type", "Unknown message type.")
    return None


def create_app(model: BDH) -> FastAPI:
    torch.set_num_threads(CPU_THREADS)
    app = FastAPI(title="DragonScope", docs_url=None, redoc_url=None, openapi_url=None)
    session = DemoSession(model)
    facts = model_facts(model)
    lock = asyncio.Lock()
    clients: set[WebSocket] = set()
    app.state.session = session

    async def broadcast(payload: dict) -> None:
        for client in list(clients):
            try:
                await client.send_json(payload)
            except (RuntimeError, WebSocketDisconnect):
                clients.discard(client)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        if not _same_origin(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        async with lock:
            state = await asyncio.to_thread(session.state)
            clients.add(websocket)
            await websocket.send_json(
                {"type": "hello", "guide": guide(), "model": facts, "state": state}
            )
        try:
            while True:
                raw = await websocket.receive_text()
                await _handle(websocket, raw)
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(websocket)

    async def _handle(websocket: WebSocket, raw: str) -> None:
        try:
            if len(raw) > MAX_MESSAGE_BYTES:
                raise SessionError("too_long", "That message is too long.")
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                raise SessionError("bad_json", "That message could not be read.") from None
            if not isinstance(message, dict):
                raise SessionError("bad_json", "That message could not be read.")

            # Generation and rebuilds are slow, so keep the event loop free. The
            # lock keeps clients from changing the session during either one.
            async with lock:
                result = await asyncio.to_thread(_dispatch, session, message)
                state = await asyncio.to_thread(session.state)
                activity = session.take_activity()
                if activity is not None:
                    await broadcast({"type": "activity", "frames": activity})
                if result is not None:
                    await broadcast(result)
                await broadcast(state)
        except SessionError as error:
            await websocket.send_json(
                {"type": "error", "code": error.code, "message": error.message}
            )
        except WebSocketDisconnect:
            raise
        except Exception as error:
            # Only the type: a traceback or message could include chat content.
            logger.error("request failed: %s", type(error).__name__)
            await websocket.send_json(
                {"type": "error", "code": "internal", "message": "Something went wrong."}
            )

    # Last, so that it does not shadow the routes above.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = create_app(load_model(args.model))
    print(f"DragonScope is running at http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        log_level="warning",
        ws_max_size=MAX_MESSAGE_BYTES * 4,
    )


if __name__ == "__main__":
    main()
