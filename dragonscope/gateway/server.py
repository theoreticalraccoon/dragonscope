"""The review console: a JSON API and one page, on 127.0.0.1 only.

Access logs are off, errors never repeat what was submitted, and requests from
other web pages are refused, as in the booth server.
"""

from __future__ import annotations

import asyncio
import json
import logging
from random import Random

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from dragonscope.gateway import samples
from dragonscope.gateway.pipeline import Gateway, GatewayError
from dragonscope.server import STATIC_DIR, UncachedFiles, _same_origin

logger = logging.getLogger("dragonscope.gateway")
MAX_REQUEST_BYTES = 16_384


def create_app(gateway: Gateway) -> FastAPI:
    app = FastAPI(title="DragonScope gateway", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.gateway = gateway
    rng = Random()

    async def body(request: Request) -> dict:
        # Same-origin and JSON only, so another page cannot post here.
        if not _same_origin(request):
            raise GatewayError("Requests from other pages are refused.")
        if request.headers.get("content-type", "").split(";")[0] != "application/json":
            raise GatewayError("Send JSON.")
        raw = await request.body()
        if len(raw) > MAX_REQUEST_BYTES:
            raise GatewayError("That request is too large.")
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            raise GatewayError("That request could not be read.") from None
        if not isinstance(message, dict):
            raise GatewayError("That request could not be read.")
        return message

    def refused(error: Exception) -> JSONResponse:
        if isinstance(error, GatewayError):
            status = 403 if "other pages" in str(error) else 400
            return JSONResponse({"error": str(error)}, status_code=status)
        # Only the type: a message could include case content.
        logger.error("request failed: %s", type(error).__name__)
        return JSONResponse({"error": "Something went wrong."}, status_code=500)

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(STATIC_DIR / "gateway.html", headers={"Cache-Control": "no-store"})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/state")
    def state() -> dict:
        return gateway.state()

    @app.get("/api/sample/{kind}")
    def sample(kind: str) -> JSONResponse:
        makers = {"bank": samples.bank_case, "clinical": samples.clinical_case}
        if kind not in makers:
            return JSONResponse({"error": "Unknown sample."}, status_code=404)
        made = makers[kind](rng)
        return JSONResponse({"text": made.text, "task": made.task})

    @app.post("/api/submit")
    async def submit(request: Request) -> JSONResponse:
        try:
            message = await body(request)
            ref = message.get("case_ref")
            text = message.get("case_text")
            if isinstance(text, str) and text.strip():
                await asyncio.to_thread(gateway.open_case, ref, text)
            draft = await asyncio.to_thread(
                gateway.submit, ref, message.get("task"), message.get("requester")
            )
            return JSONResponse({"draft": draft.to_dict()})
        except Exception as error:
            return refused(error)

    @app.post("/api/review")
    async def review(request: Request) -> JSONResponse:
        try:
            message = await body(request)
            draft_id, decision = message.get("draft"), message.get("decision")
            if not isinstance(draft_id, int) or decision not in ("approve", "reject"):
                raise GatewayError("Choose a draft and approve or reject it.")
            draft = await asyncio.to_thread(
                gateway.decide, draft_id, message.get("reviewer"), decision == "approve"
            )
            return JSONResponse({"draft": draft.to_dict()})
        except Exception as error:
            return refused(error)

    @app.post("/api/forget")
    async def forget(request: Request) -> JSONResponse:
        try:
            message = await body(request)
            result = await asyncio.to_thread(
                gateway.forget_fact, message.get("case_ref"), message.get("value")
            )
            return JSONResponse({"forget": result})
        except Exception as error:
            return refused(error)

    @app.post("/api/erase")
    async def erase(request: Request) -> JSONResponse:
        try:
            message = await body(request)
            await asyncio.to_thread(gateway.erase_case, message.get("case_ref"))
            return JSONResponse({"erased": True})
        except Exception as error:
            return refused(error)

    app.mount("/static", UncachedFiles(directory=STATIC_DIR, html=False), name="static")
    return app
