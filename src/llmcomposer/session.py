"""Stateful composing sessions built on the composer agent."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.models import Model

from .agent import DEFAULT_MODEL, CompositionDeps, composer_agent
from .models import ScoreUpdate, TurnMeta


def resolve_model() -> str:
    """Return the configured model name.

    Reads ``LLMCOMPOSER_MODEL`` and falls back to :data:`DEFAULT_MODEL`.
    Set ``LLMCOMPOSER_MODEL=offline`` to use the built-in offline composer.
    """
    return os.environ.get("LLMCOMPOSER_MODEL", DEFAULT_MODEL)


class ComposerSession:
    """One evolving collaboration: typed message history plus the score.

    Parameters
    ----------
    model : Model | str | None
        The model to run the agent with. ``None`` resolves from the
        environment via :func:`resolve_model`; the string ``"offline"``
        selects the built-in deterministic composer.
    """

    def __init__(self, model: Model | str | None = None) -> None:
        resolved = model if model is not None else resolve_model()
        if resolved == "offline":
            from .offline import offline_model

            resolved = offline_model()
        self._model: Model | str = resolved
        self._history: list[ModelMessage] = []
        self.abc: str | None = None

    @property
    def model_name(self) -> str:
        """A human-readable label for the session's model."""
        model = self._model
        if isinstance(model, Model):
            return f"{model.system}:{model.model_name}"
        return model

    @property
    def history(self) -> list[ModelMessage]:
        """The typed pydantic-ai message history accumulated so far."""
        return list(self._history)

    async def send(self, message: str) -> tuple[ScoreUpdate, TurnMeta]:
        """Send one chat message to the composer and update the score.

        Parameters
        ----------
        message : str
            The collaborator's natural-language request.

        Returns
        -------
        tuple[ScoreUpdate, TurnMeta]
            The composer's validated reply and complete updated score, plus
            transparency data about the turn (usage, retries, timing).
        """
        started = time.monotonic()
        result = await composer_agent.run(
            message,
            model=self._model,
            deps=CompositionDeps(current_abc=self.abc),
            message_history=self._history,
        )
        return self._finish(result, started)

    async def send_stream(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """Send a message and yield live progress events, then the result.

        Yields dicts suitable for server-sent events:

        - ``{"type": "writing", "attempt": n}`` — the model started (or, on
          a validator bounce, restarted) writing the score.
        - ``{"type": "progress", "chars": n}`` — score text streamed so far.
        - ``{"type": "final", "reply": ..., "abc": ..., "meta": {...}}``

        Parameters
        ----------
        message : str
            The collaborator's natural-language request.
        """
        started = time.monotonic()
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        attempts = 0
        chars = 0
        last_reported = 0

        async def handler(
            _ctx: RunContext[CompositionDeps],
            events: AsyncIterable[AgentStreamEvent],
        ) -> None:
            nonlocal attempts, chars, last_reported
            async for event in events:
                if isinstance(event, PartStartEvent) and isinstance(
                    event.part, ToolCallPart
                ):
                    attempts += 1
                    chars = 0
                    last_reported = 0
                    await queue.put({"type": "writing", "attempt": attempts})
                elif isinstance(event, PartDeltaEvent) and isinstance(
                    event.delta, ToolCallPartDelta
                ):
                    delta = event.delta.args_delta
                    chars += len(delta) if isinstance(delta, str) else 0
                    if chars - last_reported >= 120:
                        last_reported = chars
                        await queue.put({"type": "progress", "chars": chars})

        async def run() -> tuple[ScoreUpdate, TurnMeta]:
            try:
                result = await composer_agent.run(
                    message,
                    model=self._model,
                    deps=CompositionDeps(current_abc=self.abc),
                    message_history=self._history,
                    event_stream_handler=handler,
                )
                return self._finish(result, started)
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        while (event := await queue.get()) is not None:
            yield event
        update, meta = await task
        yield {
            "type": "final",
            "reply": update.reply,
            "abc": update.abc,
            "meta": meta.model_dump(),
        }

    def _finish(
        self, result: AgentRunResult[ScoreUpdate], started: float
    ) -> tuple[ScoreUpdate, TurnMeta]:
        """Record a finished run: history, score, and transparency data."""
        corrections = sum(
            isinstance(part, RetryPromptPart)
            for message_ in result.new_messages()
            for part in message_.parts
        )
        usage = result.usage
        meta = TurnMeta(
            model=self.model_name,
            requests=usage.requests,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            corrections=corrections,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        self._history = list(result.all_messages())
        self.abc = result.output.abc
        return result.output, meta

    def reset(self) -> None:
        """Forget the score and the conversation; start from silence."""
        self._history = []
        self.abc = None
