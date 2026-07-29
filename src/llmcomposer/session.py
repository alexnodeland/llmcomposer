"""Stateful composing sessions built on the composer agent."""

from __future__ import annotations

import os
import time

from pydantic_ai.messages import ModelMessage, RetryPromptPart
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
        corrections = sum(
            isinstance(part, RetryPromptPart)
            for message_ in result.new_messages()
            for part in message_.parts
        )
        usage = result.usage()
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
