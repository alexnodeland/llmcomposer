"""Typed domain models shared across the llmcomposer package."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n|\n?\s*```\s*$")


class ScoreUpdate(BaseModel):
    """The composer agent's structured turn: a reply and the full score.

    This is the agent's ``output_type`` — every turn the model must return
    both a short message for the collaborator and the *complete* updated
    tune, never a fragment or a diff.
    """

    reply: str = Field(
        description=(
            "A short, warm message to your collaborator describing what you "
            "wrote or changed and why. One to three sentences, lowercase, "
            "no markdown."
        )
    )
    abc: str = Field(
        description=(
            "The complete updated tune in ABC notation. Must start with the "
            "X: header and include M:, L:, and K: fields plus the full tune "
            "body. Raw ABC only — no markdown fences, no commentary."
        )
    )

    @field_validator("abc")
    @classmethod
    def _strip_markdown_fences(cls, value: str) -> str:
        """Remove accidental markdown fences before deeper validation runs."""
        previous = None
        while previous != value:
            previous = value
            value = _FENCE.sub("", value)
        return value.strip()


class ChatRequest(BaseModel):
    """A collaborator's chat message posted to the web app."""

    message: str = Field(min_length=1, max_length=4000)


class TurnMeta(BaseModel):
    """Transparency data for one composing turn.

    Surfaced in the UI so the collaboration is inspectable: which model ran,
    how many requests it took, token spend, how often the score was bounced
    back by the ABC validator, and wall-clock time.
    """

    model: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    corrections: int = 0
    elapsed_ms: int = 0


class ChatResponse(BaseModel):
    """The web app's answer: reply, current score, and turn transparency."""

    reply: str
    abc: str
    meta: TurnMeta
