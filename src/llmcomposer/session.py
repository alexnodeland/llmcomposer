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

    Reads ``LLMCOMPOSER_MODEL``, then the first entry of
    ``LLMCOMPOSER_MODELS``, and falls back to :data:`DEFAULT_MODEL`. Set
    ``LLMCOMPOSER_MODEL=offline`` to use the built-in offline composer.
    """
    configured = os.environ.get("LLMCOMPOSER_MODEL")
    if configured:
        return configured
    models = os.environ.get("LLMCOMPOSER_MODELS", "")
    first = models.split(",")[0].strip()
    return first or DEFAULT_MODEL


def available_models() -> list[str]:
    """Return the model names offered in the studio's model selector.

    Reads ``LLMCOMPOSER_MODELS`` (comma-separated) when set, otherwise the
    single model from :func:`resolve_model`. The ``offline`` composer is
    always included so the studio works with no credentials.
    """
    configured = os.environ.get("LLMCOMPOSER_MODELS")
    if configured:
        names = [name.strip() for name in configured.split(",") if name.strip()]
    else:
        names = [resolve_model()]
    if "offline" not in names:
        names.append("offline")
    return names


def build_model(name: str) -> tuple[Model | str, str]:
    """Resolve a configured model name into a runnable model plus a label.

    ``offline`` selects the built-in deterministic composer. Names prefixed
    ``litellm:`` are routed through a LiteLLM proxy configured with
    ``LITELLM_BASE_URL`` and ``LITELLM_API_KEY``. Anything else is passed
    through as a pydantic-ai model name (``anthropic:claude-opus-5``, …).
    """
    if name == "offline":
        from .offline import offline_model

        model = offline_model()
        return model, f"{model.system}:{model.model_name}"
    if name.startswith("litellm:"):
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.litellm import LiteLLMProvider

        model = OpenAIChatModel(
            name.removeprefix("litellm:"),
            provider=LiteLLMProvider(
                api_key=os.environ.get("LITELLM_API_KEY"),
                api_base=os.environ.get("LITELLM_BASE_URL"),
            ),
        )
        return model, name
    return name, name


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
        if isinstance(model, Model):
            self._model: Model | str = model
            self._label = f"{model.system}:{model.model_name}"
        else:
            self._model, self._label = build_model(
                model if model is not None else resolve_model()
            )
        self._history: list[ModelMessage] = []
        self.abc: str | None = None

    @property
    def model_name(self) -> str:
        """A human-readable label for the session's model."""
        return self._label

    def set_model(self, name: str) -> None:
        """Switch the session to another model, keeping history and score.

        The conversation and the working score carry over — the new model
        picks up the same collaboration mid-stream.
        """
        self._model, self._label = build_model(name)

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
