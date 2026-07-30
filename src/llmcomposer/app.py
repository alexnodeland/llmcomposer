"""FastAPI application serving the llmcomposer collaborative chat."""

from __future__ import annotations

import asyncio
import json
import secrets
from collections import OrderedDict
from collections.abc import AsyncIterator
from pathlib import Path

import logfire
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic_ai.models import Model

from .abc_notation import ABCValidationError, validate_abc
from .models import ChatRequest, ChatResponse, ModelSelectRequest, ScoreRequest
from .session import ComposerSession, available_models

_PACKAGE = Path(__file__).parent
_TEMPLATES = Jinja2Templates(directory=str(_PACKAGE / "templates"))

SESSION_COOKIE = "llmc_session"
"""Cookie carrying the opaque id of the caller's composing session."""

MAX_SESSIONS = 64
"""How many sessions the studio keeps before evicting the least recent."""


class SessionStore:
    """A small LRU of composing sessions, each with its own lock.

    One studio can hold several conversations at once — two browser tabs,
    two people, two eval workers — without them writing over each other's
    score. Keys are opaque and only ever minted here, so a caller cannot
    seat itself at a session id of its own choosing.

    Parameters
    ----------
    model : Model | str | None
        Model override handed to every session it creates.
    capacity : int
        Maximum number of live sessions; the least recently used is
        dropped when a new one would exceed it.
    """

    def __init__(
        self, model: Model | str | None = None, capacity: int = MAX_SESSIONS
    ) -> None:
        self._model = model
        self._capacity = capacity
        self._seats: OrderedDict[str, tuple[ComposerSession, asyncio.Lock]] = (
            OrderedDict()
        )

    def __len__(self) -> int:
        """Return the number of live sessions."""
        return len(self._seats)

    def seat(self, key: str | None) -> tuple[str, ComposerSession, asyncio.Lock]:
        """Return the session for ``key``, opening a fresh one if needed.

        Parameters
        ----------
        key : str | None
            The id read from the caller's cookie, if any.

        Returns
        -------
        tuple[str, ComposerSession, asyncio.Lock]
            The (possibly newly minted) id, its session, and its lock.
        """
        if key is not None and key in self._seats:
            self._seats.move_to_end(key)
            session, lock = self._seats[key]
            return key, session, lock
        fresh = secrets.token_urlsafe(18)
        session, lock = ComposerSession(model=self._model), asyncio.Lock()
        self._seats[fresh] = (session, lock)
        while len(self._seats) > self._capacity:
            self._seats.popitem(last=False)
        return fresh, session, lock


def _issue(response: Response, key: str) -> None:
    """Hand the caller back the cookie naming their session."""
    response.set_cookie(
        SESSION_COOKIE, key, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 7
    )


def create_app(model: Model | str | None = None) -> FastAPI:
    """Build the web app around a store of per-caller composing sessions.

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
    app.mount("/static", StaticFiles(directory=_PACKAGE / "static"), name="static")
    store = SessionStore(model=model)

    def seat(request: Request) -> tuple[str, ComposerSession, asyncio.Lock]:
        return store.seat(request.cookies.get(SESSION_COOKIE))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        key, session, _ = seat(request)
        page = _TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {"model_name": session.model_name, "models": available_models()},
        )
        _issue(page, key)
        return page

    @app.get("/models")
    async def models(request: Request, response: Response) -> dict[str, object]:
        key, session, _ = seat(request)
        _issue(response, key)
        return {"models": available_models(), "current": session.model_name}

    @app.post("/model")
    async def select_model(
        payload: ModelSelectRequest, request: Request, response: Response
    ) -> dict[str, str]:
        if payload.model not in available_models():
            raise HTTPException(status_code=404, detail="unknown model")
        key, session, lock = seat(request)
        async with lock:
            session.set_model(payload.model)
        _issue(response, key)
        return {"model": session.model_name}

    @app.post("/score")
    async def set_score(
        payload: ScoreRequest, request: Request, response: Response
    ) -> dict[str, bool]:
        try:
            validate_abc(payload.abc)
        except ABCValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        key, session, lock = seat(request)
        async with lock:
            session.abc = payload.abc
        _issue(response, key)
        return {"ok": True}

    @app.post("/chat")
    async def chat(
        payload: ChatRequest, request: Request, response: Response
    ) -> ChatResponse:
        key, session, lock = seat(request)
        async with lock:
            try:
                update, meta = await session.send(payload.message)
            except Exception as exc:
                logfire.exception("chat turn failed")
                raise HTTPException(status_code=502, detail=str(exc)) from exc
        _issue(response, key)
        return ChatResponse(reply=update.reply, abc=update.abc, meta=meta)

    @app.post("/chat/stream")
    async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
        key, session, lock = seat(request)

        async def events() -> AsyncIterator[str]:
            async with lock:
                try:
                    async for event in session.send_stream(payload.message):
                        yield f"data: {json.dumps(event)}\n\n"
                except Exception as exc:  # noqa: BLE001 - the stream is open
                    # Headers went out before the model ran, so a dropped
                    # provider would otherwise close the connection in
                    # silence. One last frame is all we have left to say it.
                    logfire.exception("chat stream failed")
                    error = {"type": "error", "detail": str(exc)}
                    yield f"data: {json.dumps(error)}\n\n"

        stream = StreamingResponse(events(), media_type="text/event-stream")
        _issue(stream, key)
        return stream

    @app.post("/reset")
    async def reset(request: Request, response: Response) -> dict[str, bool]:
        key, session, lock = seat(request)
        async with lock:
            session.reset()
        _issue(response, key)
        return {"ok": True}

    return app


def main() -> None:
    """Run the development server (``uv run llmcomposer``)."""
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
