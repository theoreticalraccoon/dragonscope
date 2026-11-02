"""A live dashboard for any BDH session.

Run: python -m dragonscope.dashboard --model models/best_model.pt
Or, from Python: ds.dashboard(memory)

It binds to 127.0.0.1 and keeps everything in memory. Access logs are off, and
nothing here logs message contents.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from dragonscope.activity import ActivityRecorder, model_facts
from dragonscope.engine import SessionMemory
from dragonscope.model_config import demo_config
from dragonscope.report import Case, Probe, forgetting_report
from dragonscope.server import MAX_MESSAGE_BYTES, STATIC_DIR, UncachedFiles, _same_origin
from dragonscope.toolkit import MemoryCapture, capture
from dragonscope.train import load_model
from dragonscope.vendor.bdh.bdh import BDH

logger = logging.getLogger("dragonscope.dashboard")
# Time each byte is held on screen, so the activity can be watched.
PACE = 0.04
# The ledger grows with every byte, so this keeps a stray paste from filling memory.
MAX_POSITION = 2000
MAX_GENERATE = 200


class InspectorError(Exception):
    """A request the dashboard refused. Messages never repeat what was submitted."""


class Inspector:
    """Connects a ``MemoryCapture`` to the page, reporting each byte as it is computed."""

    def __init__(self, memory: MemoryCapture, pace: float = PACE):
        self.capture = memory
        self.pace = pace
        self.emit = None
        self.watched: list[tuple[int, int]] = []
        self.last_edit: dict | None = None
        self.report: dict | None = None
        self.recorder = ActivityRecorder(memory.model, memory.memory.n_layer)
        memory.add_observer(self.recorder)

    # Watching

    def _watched_positions(self) -> set[int]:
        engine = self.capture.memory
        return {
            position
            for start, end in self.watched
            for position in range(start, end)
            if position in engine.bytes
        }

    def _counterfactual(self) -> SessionMemory | None:
        positions = self._watched_positions()
        if not positions:
            return None
        shadow = self.capture.memory.fork()
        for position in positions:
            shadow.forget(position, position + 1, "subtract")
        shadow.rebuild()
        shadow.observer = self.recorder.watch_counterfactual
        return shadow

    def watch(self, text: str) -> None:
        spans = self.capture.find(text)
        if not spans:
            raise InspectorError("That text is not in memory.")
        self.watched += [span for span in spans if span not in self.watched]

    # Reading and writing

    def _step(self, byte: int, phase: str, shadow: SessionMemory | None) -> None:
        engine = self.capture.memory
        chosen_from = engine.next_logits
        without_from = None if shadow is None else shadow.next_logits
        started = time.perf_counter()
        if shadow is not None:
            shadow.read(bytes([byte]))
        self.capture.read(bytes([byte]))
        computed = time.perf_counter() - started

        depends = 0.0
        if phase == "write" and without_from is not None and chosen_from is not None:
            depends = round(
                max(
                    0.0,
                    float(
                        chosen_from.softmax(dim=0)[byte] - without_from.softmax(dim=0)[byte]
                    ),
                ),
                3,
            )
        frame = self.recorder.take_frame()
        frame.update(
            type="frame",
            phase=phase,
            char=chr(byte) if byte < 128 else "�",
            depends=depends,
            ms=round(computed * 1000, 1),
            show_ms=round(max(computed, self.pace) * 1000),
        )
        if self.emit is not None:
            self.emit(frame)
        if self.pace > computed:
            time.sleep(self.pace - computed)

    def _check_room(self, count: int) -> None:
        if self.capture.position + count > MAX_POSITION:
            raise InspectorError("This session is full. Please reset.")

    def read(self, text: str) -> None:
        data = text.encode("utf-8")
        if not data:
            raise InspectorError("There is nothing to read.")
        self._check_room(len(data))
        shadow = self._counterfactual()
        for byte in data:
            self._step(byte, "read", shadow)

    def generate(self, count: int, stop: str | None = "\n") -> None:
        engine = self.capture.memory
        if engine.next_logits is None:
            raise InspectorError("Read something first. Forgetting clears the pending output.")
        count = max(1, min(int(count), MAX_GENERATE))
        self._check_room(count)
        shadow = self._counterfactual()
        stop_byte = None if not stop else stop.encode("utf-8")[0]
        for _ in range(count):
            byte = int(engine.next_logits.argmax())
            self._step(byte, "write", shadow)
            if byte == stop_byte:
                break

    # Editing

    def forget(self, method: str, text: str | None, start: int | None, end: int | None) -> None:
        target = text if text else (start, end)
        if not text and not (isinstance(start, int) and isinstance(end, int) and start < end):
            raise InspectorError("Choose some text to forget.")
        try:
            result = self.capture.forget(target, method)
        except ValueError:
            raise InspectorError("That text is not in memory.") from None
        if not result.bytes_erased:
            raise InspectorError("That text is not in memory.")
        self.last_edit = {
            "method": result.method,
            "copies": result.copies,
            "bytes_erased": result.bytes_erased,
            "ms": round(result.seconds * 1000, 1),
            "residual": result.residual,
            "residual_per_round": result.residual_per_round,
        }

    def rebuild(self) -> None:
        started = time.perf_counter()
        self.capture.rebuild()
        seconds = time.perf_counter() - started
        residual = self.capture.residual()
        self.last_edit = {
            "method": "rebuild",
            "copies": 0,
            "bytes_erased": 0,
            "ms": round(seconds * 1000, 1),
            "residual": residual["total"],
            "residual_per_round": residual["per_round"],
        }

    def reset(self) -> None:
        self.capture.reset()
        self.watched = []
        self.last_edit = None
        self.report = None

    def run_report(self, forget: str, probes: list[dict]) -> None:
        """Test forgetting on a copy of the surviving text, leaving this session alone."""
        text = self.capture.text
        if not text:
            raise InspectorError("Read something first.")
        if not forget or forget not in text:
            raise InspectorError("The text to forget is not in memory.")
        parsed = []
        for probe in probes:
            prompt, expect = probe.get("prompt"), probe.get("expect")
            about = probe.get("about")
            if not prompt or not expect or about not in ("forgotten", "kept"):
                raise InspectorError("Each probe needs a prompt and expected text.")
            parsed.append(Probe(prompt, expect, about))
        if not parsed:
            raise InspectorError("Add at least one probe.")
        report = forgetting_report(
            self.capture.model, [Case(text, [forget], parsed)]
        ).to_dict()
        # Replies can repeat a value, and this report outlives the session's forgetting.
        report.pop("samples", None)
        self.report = report

    # What the page sees

    def state(self) -> dict:
        engine = self.capture.memory
        runs: list[dict] = []
        for position, byte in sorted(engine.bytes.items()):
            character = chr(byte) if byte < 128 else "�"
            if runs and runs[-1]["start"] + len(runs[-1]["text"]) == position:
                runs[-1]["text"] += character
            else:
                runs.append({"start": position, "text": character})

        strength = engine.rho.norm(dim=(0, 2)).flatten()
        scaled = strength / strength.max().clamp_min(1e-12)
        share = torch.zeros_like(strength)
        positions = self._watched_positions()
        if positions:
            held = torch.zeros_like(engine.rho)
            for position in positions:
                entry = engine.ledger[position]
                for level in range(engine.n_layer):
                    held[level] += torch.einsum("d,hn->hdn", entry.v[level], entry.qr[level])
            share = (held.norm(dim=(0, 2)).flatten() / strength.clamp_min(1e-12)).clamp_max(1.0)
        ledger_entries = len(engine.ledger)
        ledger_bytes = 0
        if engine.ledger:
            entry = next(iter(engine.ledger.values()))
            per_byte = entry.v.numel() * entry.v.element_size() + entry.qr.numel() * entry.qr.element_size()
            ledger_bytes = per_byte * ledger_entries
        return {
            "type": "state",
            "runs": runs,
            "watched": [
                [start, end]
                for start, end in self.watched
                if any(p in engine.bytes for p in range(start, end))
            ],
            "can_generate": engine.next_logits is not None,
            "last_edit": self.last_edit,
            "report": self.report,
            "memory": {
                "position": engine.position,
                "position_limit": MAX_POSITION,
                "bytes_stored": len(engine.bytes),
                "ledger_entries": ledger_entries,
                "ledger_mb": round(ledger_bytes / 1e6, 1),
                "rho_mb": round(engine.rho.numel() * engine.rho.element_size() / 1e6, 1),
                "rho_norm": round(float(engine.rho.norm()), 3),
                "steps_recorded": len(self.capture.history),
                "strength": (scaled * 100).round().int().tolist(),
                "flagged": (share * 100).round().int().tolist(),
            },
        }

    def dispatch(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "read":
            text = message.get("text")
            if not isinstance(text, str):
                raise InspectorError("There is nothing to read.")
            self.read(text)
        elif kind == "generate":
            count = message.get("count", 70)
            if not isinstance(count, int):
                raise InspectorError("Choose how many bytes to write.")
            self.generate(count)
        elif kind == "watch":
            text = message.get("text")
            if not isinstance(text, str) or not text:
                raise InspectorError("Choose some text to watch.")
            self.watch(text)
        elif kind == "unwatch":
            self.watched = []
        elif kind == "forget":
            method = message.get("method", "subtract")
            if method not in ("subtract", "rebuild"):
                raise InspectorError("Unknown method.")
            text = message.get("text")
            self.forget(
                method,
                text if isinstance(text, str) else None,
                message.get("start"),
                message.get("end"),
            )
        elif kind == "rebuild":
            self.rebuild()
        elif kind == "report":
            forget = message.get("forget")
            probes = message.get("probes")
            if not isinstance(forget, str) or not isinstance(probes, list):
                raise InspectorError("The report needs text to forget and some probes.")
            if not all(isinstance(probe, dict) for probe in probes):
                raise InspectorError("Each probe needs a prompt and expected text.")
            self.run_report(forget, probes)
        elif kind == "reset":
            self.reset()
        else:
            raise InspectorError("Unknown message type.")


def create_dashboard(memory: MemoryCapture, pace: float = PACE) -> FastAPI:
    app = FastAPI(title="DragonScope", docs_url=None, redoc_url=None, openapi_url=None)
    inspector = Inspector(memory, pace)
    facts = model_facts(memory.model)
    lock = asyncio.Lock()
    clients: set[WebSocket] = set()
    app.state.inspector = inspector

    async def broadcast(payload: dict) -> None:
        for client in list(clients):
            try:
                await client.send_json(payload)
            except (RuntimeError, ConnectionError, WebSocketDisconnect):
                clients.discard(client)

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(STATIC_DIR / "inspector.html", headers={"Cache-Control": "no-store"})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    async def handle(websocket: WebSocket, raw: str) -> None:
        try:
            if len(raw) > MAX_MESSAGE_BYTES:
                raise InspectorError("That message is too long.")
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                raise InspectorError("That message could not be read.") from None
            if not isinstance(message, dict):
                raise InspectorError("That message could not be read.")
            async with lock:
                loop = asyncio.get_running_loop()
                outgoing: asyncio.Queue = asyncio.Queue()

                async def send_frames() -> None:
                    while (frame := await outgoing.get()) is not None:
                        await broadcast(frame)

                inspector.emit = lambda frame: loop.call_soon_threadsafe(
                    outgoing.put_nowait, frame
                )
                sender = asyncio.create_task(send_frames())
                try:
                    await asyncio.to_thread(inspector.dispatch, message)
                finally:
                    outgoing.put_nowait(None)
                    await sender
                    inspector.emit = None
                await broadcast(await asyncio.to_thread(inspector.state))
        except InspectorError as error:
            await websocket.send_json({"type": "error", "message": str(error)})
        except WebSocketDisconnect:
            raise
        except Exception as error:
            # Only the type: a traceback or message could include session content.
            logger.error("request failed: %s", type(error).__name__)
            await websocket.send_json({"type": "error", "message": "Something went wrong."})

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        if not _same_origin(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        async with lock:
            state = await asyncio.to_thread(inspector.state)
            clients.add(websocket)
            await websocket.send_json({"type": "hello", "model": facts, "state": state})
        try:
            while True:
                await handle(websocket, await websocket.receive_text())
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(websocket)

    app.mount("/", UncachedFiles(directory=STATIC_DIR, html=False), name="static")
    return app


def _server(memory: MemoryCapture, port: int, pace: float) -> uvicorn.Server:
    return uvicorn.Server(
        uvicorn.Config(
            create_dashboard(memory, pace),
            host="127.0.0.1",
            port=port,
            access_log=False,
            log_level="warning",
            ws_per_message_deflate=False,
            ws_max_size=MAX_MESSAGE_BYTES * 4,
        )
    )


def dashboard(memory: MemoryCapture, port: int = 8100, pace: float = PACE) -> str:
    """Open a session in the dashboard, in the background, and return its address."""
    server = _server(memory, port, pace)
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", type=Path, help="a saved model, from dragonscope.train")
    source.add_argument("--untrained", action="store_true", help="try the dashboard with random weights")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    if args.untrained:
        torch.manual_seed(0)
        model: BDH = BDH(demo_config()).eval()
    else:
        model = load_model(args.model or Path("models/best_model.pt"))
    print(f"DragonScope dashboard: http://127.0.0.1:{args.port}", flush=True)
    _server(capture(model), args.port, PACE).run()


if __name__ == "__main__":
    main()
