"""Stateful composing sessions built on the composer agent."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    PartDeltaEvent,
    PartStartEvent,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from .agent import (
    DEFAULT_MODEL,
    PROMPT_SHA,
    PROMPT_VERSION,
    CompositionDeps,
    composer_agent,
)
from .models import ScoreUpdate, TurnMeta
from .recording import Recorder


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


def _env_number(name: str, cast: type[float | int]) -> float | int | None:
    """Read an optional numeric environment variable, ignoring junk."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return cast(raw)
    except ValueError:
        return None


def resolve_sampling(
    temperature: float | None = None,
    seed: int | None = None,
) -> tuple[ModelSettings | None, float | None, int | None]:
    """Resolve the reproducibility knobs, explicit arguments first.

    Falls back to ``LLMCOMPOSER_TEMPERATURE`` and ``LLMCOMPOSER_SEED``
    for whichever argument is ``None``. Everything is optional; an unset
    or unparseable value leaves the provider's default in place. The
    resolved values travel in :class:`~llmcomposer.models.TurnMeta` so a
    recorded score can be tied to the settings that produced it.

    Parameters
    ----------
    temperature : float | None, optional
        An explicit sampling temperature. Wins over the environment.
    seed : int | None, optional
        An explicit sampling seed. Wins over the environment.

    Returns
    -------
    tuple[ModelSettings | None, float | None, int | None]
        Settings to pass to the agent (``None`` when nothing is
        configured), the resolved temperature, and the resolved seed.
    """
    chosen_temperature = (
        temperature
        if temperature is not None
        else _env_number("LLMCOMPOSER_TEMPERATURE", float)
    )
    chosen_seed = seed if seed is not None else _env_number("LLMCOMPOSER_SEED", int)
    settings = ModelSettings()
    if chosen_temperature is not None:
        settings["temperature"] = float(chosen_temperature)
    if chosen_seed is not None:
        settings["seed"] = int(chosen_seed)
    if not settings:
        return None, None, None
    return (
        settings,
        float(chosen_temperature) if chosen_temperature is not None else None,
        int(chosen_seed) if chosen_seed is not None else None,
    )


class ComposerSession:
    """One evolving collaboration: typed message history plus the score.

    Parameters
    ----------
    model : Model | str | None
        The model to run the agent with. ``None`` resolves from the
        environment via :func:`resolve_model`; the string ``"offline"``
        selects the built-in deterministic composer.
    recorder : Recorder | None
        Sink for finished turns. ``None`` builds one that defers to
        ``LLMCOMPOSER_RECORD_DIR`` and stays a no-op while it is unset.
    temperature : float | None
        Sampling temperature for every turn of this session. ``None``
        defers to ``LLMCOMPOSER_TEMPERATURE``; an explicit value wins over
        it, so a sweep can pin sampling per cell without touching the
        process environment.
    seed : int | None
        Sampling seed, resolved the same way against ``LLMCOMPOSER_SEED``.
    """

    def __init__(
        self,
        model: Model | str | None = None,
        recorder: Recorder | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> None:
        if isinstance(model, Model):
            self._model: Model | str = model
            self._label = f"{model.system}:{model.model_name}"
        else:
            self._model, self._label = build_model(
                model if model is not None else resolve_model()
            )
        self._history: list[ModelMessage] = []
        self.abc: str | None = None
        self.session_id = uuid.uuid4().hex
        self._recorder = recorder if recorder is not None else Recorder()
        self._turn_index = 0
        self._temperature = temperature
        self._seed = seed

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
        settings, temperature, seed = resolve_sampling(self._temperature, self._seed)
        deps = CompositionDeps(current_abc=self.abc)
        result = await composer_agent.run(
            message,
            model=self._model,
            deps=deps,
            message_history=self._history,
            model_settings=settings,
        )
        return self._finish(message, deps, result, started, temperature, seed)

    async def send_stream(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """Send a message and yield live progress events, then the result.

        Yields dicts suitable for server-sent events:

        - ``{"type": "bounce", "attempt": n, "code": ..., "reason": ...}`` —
          the validator refused attempt ``n``, and here is why.
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
        settings, temperature, seed = resolve_sampling(self._temperature, self._seed)
        deps = CompositionDeps(current_abc=self.abc)
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
                    # A rewrite means the validator just refused the previous
                    # attempt; say what it objected to before writing resumes.
                    if attempts > 1 and deps.bounces:
                        bounce = deps.bounces[-1]
                        await queue.put(
                            {
                                "type": "bounce",
                                "attempt": attempts - 1,
                                "code": bounce.code,
                                "reason": bounce.message,
                            }
                        )
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
                    deps=deps,
                    message_history=self._history,
                    model_settings=settings,
                    event_stream_handler=handler,
                )
                return self._finish(message, deps, result, started, temperature, seed)
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
        self,
        message: str,
        deps: CompositionDeps,
        result: AgentRunResult[ScoreUpdate],
        started: float,
        temperature: float | None,
        seed: int | None,
    ) -> tuple[ScoreUpdate, TurnMeta]:
        """Close out a finished run: history, score, meta, and recording."""
        bounces = list(deps.bounces)
        usage = result.usage
        meta = TurnMeta(
            model=self.model_name,
            prompt_version=PROMPT_VERSION,
            prompt_sha=PROMPT_SHA,
            requests=usage.requests,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            corrections=len(bounces),
            bounces=bounces,
            temperature=temperature,
            seed=seed,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        self._history = list(result.all_messages())
        self.abc = result.output.abc
        turn_index = self._turn_index
        self._turn_index += 1
        self._recorder.record(
            session_id=self.session_id,
            turn_index=turn_index,
            user_message=message,
            reply=result.output.reply,
            abc=result.output.abc,
            meta=meta,
        )
        return result.output, meta

    def reset(self) -> None:
        """Forget the score and the conversation; start from silence.

        The session id and turn counter carry on, so recorded turns from
        before and after a reset stay distinguishable and ordered.
        """
        self._history = []
        self.abc = None
