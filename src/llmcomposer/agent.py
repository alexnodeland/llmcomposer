"""The composer: a pydantic-ai agent that writes music in ABC notation.

The agent is defined without a bound model so callers (and tests) choose one
at run time — the web app resolves it from ``LLMCOMPOSER_MODEL`` and tests
use :meth:`pydantic_ai.Agent.override` with ``TestModel``/``FunctionModel``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from .abc_notation import ABCValidationError, validate_abc
from .models import ScoreUpdate

DEFAULT_MODEL = "anthropic:claude-opus-5"
"""Model used when nothing else is configured."""

_MAX_HISTORY_MESSAGES = 24


@dataclass
class CompositionDeps:
    """Per-run dependencies: the state of the collaboration so far."""

    current_abc: str | None = None


def _keep_recent_runs(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Trim old turns while keeping run boundaries intact.

    The latest score always travels in the instructions, so old turns can be
    dropped safely. Trimming only ever starts at a ``ModelRequest`` carrying a
    ``UserPromptPart`` so tool-call/return pairs are never orphaned.
    """
    if len(messages) <= _MAX_HISTORY_MESSAGES:
        return messages
    for i in range(len(messages) - _MAX_HISTORY_MESSAGES, len(messages)):
        message = messages[i]
        if isinstance(message, ModelRequest) and any(
            isinstance(part, UserPromptPart) for part in message.parts
        ):
            return messages[i:]
    return messages


composer_agent: Agent[CompositionDeps, ScoreUpdate] = Agent(
    deps_type=CompositionDeps,
    output_type=ScoreUpdate,
    output_retries=3,
    history_processors=[_keep_recent_runs],
    instructions=(
        "You are a composer collaborating with a human through conversation. "
        "You write and revise a single tune in ABC notation; the human "
        "describes feelings, places, imagery, or concrete musical requests, "
        "and you interpret them musically.\n\n"
        "Rules for the score:\n"
        "- Always return the COMPLETE tune, never a fragment or a diff.\n"
        "- Start with X:1, then T: (an evocative lowercase title), then M:, "
        "L:, Q: and K: headers, then the tune body.\n"
        "- Keep the tune between 8 and 32 bars, playable by the abcjs synth. "
        "Guitar-style chord symbols in quotes are welcome.\n"
        "- You may write multi-instrument ARRANGEMENTS using ABC voices. "
        "Declare each voice in the header before K: as "
        'V:<n> name="<instrument>" followed immediately by a line '
        "%%MIDI program <0-127> choosing its General MIDI patch. Then write "
        "each voice's music in the body introduced by a bare V:<n> line. "
        "Choose patches with taste (e.g. 0 piano, 11 vibraphone, 24 nylon "
        "guitar, 32 acoustic bass, 40 violin, 42 cello, 46 harp, 56 trumpet, "
        "68 oboe, 71 clarinet, 73 flute). Every voice MUST have exactly the "
        "same number of bars so they play together; give accompaniment "
        "voices simpler rhythms than the melody, and low voices low "
        "registers (use , for lower octaves).\n"
        "- Use an arrangement when the human asks for instruments, a band, "
        "a duet/trio, or richer texture; keep a single voice for a simple "
        "air unless they ask otherwise.\n"
        "- Prefer musical substance over density: shape phrases, use "
        "repetition with variation, land phrases on chord tones, end on a "
        "satisfying cadence. Melodies should breathe — use rests.\n"
        "- Expression is welcome and encouraged: dynamics as decorations "
        "(!p!, !mp!, !mf!, !f!, !crescendo(!, !crescendo)!), staccato dots "
        "(.A), slurs in parentheses, and broken rhythms (A>B).\n"
        "- Craft for arrangements: give each voice its own register and "
        "role (melody sings, inner voices support with simpler rhythm, the "
        "bass moves mostly by root and fifth); avoid every voice moving in "
        "the same rhythm; let voices answer each other.\n"
        "- When revising, preserve what the human liked and change only what "
        "their message asks for, unless they ask to start over.\n\n"
        "Rules for the reply: one to three sentences, lowercase, warm and a "
        "little poetic. When composing fresh, name the concrete choices "
        "(key, mode, tempo, instruments, contour); when revising, say "
        "precisely what you changed and why it serves their request. "
        "No markdown."
    ),
)
"""The composer agent. Bind a model per run or via ``Agent.override``."""


@composer_agent.instructions
def _current_score(ctx: RunContext[CompositionDeps]) -> str:
    """Inject the working score so every turn revises the same tune."""
    if ctx.deps.current_abc:
        return (
            "The current working score is below. Revise it according to the "
            "human's message and return the complete updated tune.\n"
            f"<current_score>\n{ctx.deps.current_abc}\n</current_score>"
        )
    return (
        "There is no score yet. Compose a fresh tune (8-16 bars) that "
        "captures the human's first message."
    )


@composer_agent.output_validator
def _validated_score(
    ctx: RunContext[CompositionDeps], output: ScoreUpdate
) -> ScoreUpdate:
    """Reject malformed ABC so the model retries with a corrected score."""
    try:
        validate_abc(output.abc)
    except ABCValidationError as exc:
        raise ModelRetry(
            f"The ABC notation you returned is invalid: {exc}. Return the "
            "complete corrected tune as raw ABC."
        ) from exc
    return output
