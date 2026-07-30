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


class ModelSelectRequest(BaseModel):
    """A request to switch the session to another configured model."""

    model: str


class ScoreRequest(BaseModel):
    """A score the collaborator hands back to the studio as the working one.

    Used to restore an earlier version, or to paste in a tune from
    elsewhere. The ABC is validated before it becomes authoritative.
    """

    abc: str = Field(max_length=100_000)


class Bounce(BaseModel):
    """One validator rejection inside a single turn.

    The datum the project is built to collect: which attempt was refused,
    which error class refused it, what the validator said, and the exact
    score the model had written when it was told to try again.
    """

    attempt: int
    code: str
    message: str
    rejected_abc: str


class TurnMeta(BaseModel):
    """Transparency data for one composing turn.

    Surfaced in the UI so the collaboration is inspectable: which model ran
    under which prompt and sampling settings, how many requests it took,
    token spend, how the score was bounced back by the ABC validator, and
    wall-clock time. ``corrections`` is the length of ``bounces``.

    ``prompt_version`` is the hand-declared name of the system prompt;
    ``prompt_sha`` is its digest, so an undeclared edit to the study's
    independent variable is visible in the run log.
    """

    model: str
    prompt_version: str = ""
    prompt_sha: str = ""
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    corrections: int = 0
    bounces: list[Bounce] = []
    temperature: float | None = None
    seed: int | None = None
    elapsed_ms: int = 0


class ChatResponse(BaseModel):
    """The web app's answer: reply, current score, and turn transparency."""

    reply: str
    abc: str
    meta: TurnMeta
