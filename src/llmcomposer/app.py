"""FastAPI application serving the llmcomposer collaborative chat."""

from __future__ import annotations

import asyncio
from pathlib import Path

import logfire
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model

from .models import ChatRequest, ChatResponse
from .session import ComposerSession

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def create_app(model: Model | str | None = None) -> FastAPI:
    """Build the web app around a single composing session.

    Parameters
    ----------
    model : Model | str | None
        Model override; ``None`` resolves from ``LLMCOMPOSER_MODEL``.

    Returns
    -------
    FastAPI
        The configured application.
    """
    logfire.configure(send_to_logfire="if-token-present", console=False)
    logfire.instrument_pydantic_ai()

    app = FastAPI(title="llmcomposer")
    session = ComposerSession(model=model)
    lock = asyncio.Lock()

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request, "index.html", {"model_name": session.model_name}
        )

    @app.post("/chat")
    async def chat(payload: ChatRequest) -> ChatResponse:
        async with lock:
            try:
                update, meta = await session.send(payload.message)
            except UnexpectedModelBehavior as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
        return ChatResponse(reply=update.reply, abc=update.abc, meta=meta)

    @app.post("/reset")
    async def reset() -> dict[str, bool]:
        async with lock:
            session.reset()
        return {"ok": True}

    return app


def main() -> None:
    """Run the development server (``uv run llmcomposer``)."""
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
