"""The composer: a pydantic-ai agent that writes music in ABC notation.

The agent is defined without a bound model so callers (and tests) choose one
at run time — the web app resolves it from ``LLMCOMPOSER_MODEL`` and tests
use :meth:`pydantic_ai.Agent.override` with ``TestModel``/``FunctionModel``.

The instructions are the study's independent variable, so they live in one
versioned module-level constant (:data:`COMPOSER_INSTRUCTIONS`) stamped with
:data:`PROMPT_VERSION` and digested into :data:`PROMPT_SHA`. Every turn
records both, which is what lets a figure say *which* prompt produced it —
and catch an undeclared edit.

The instrument-range table inside the prompt is generated at import time
from :data:`llmcomposer.descriptors.GM_RANGES`, the same table the eval
harness scores range violations against, so the model is never told a bound
it will then be marked down for obeying.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from .abc_notation import ABCValidationError, validate_abc
from .descriptors import GM_RANGES
from .models import Bounce, ScoreUpdate

DEFAULT_MODEL = "anthropic:claude-opus-5"
"""Model used when nothing else is configured."""

PROMPT_VERSION = "composer-v3"
"""Identifier for the current system prompt, recorded on every turn."""

_MAX_HISTORY_MESSAGES = 24

# Sharps below the tonic-neutral spelling, flats above it: the spellings a
# player expects to read for a range bound (B flat, E flat, F sharp).
_PITCH_CLASSES = (
    "C",
    "^C",
    "D",
    "_E",
    "E",
    "F",
    "^F",
    "G",
    "_A",
    "A",
    "_B",
    "B",
)

# The instruments the prompt names, each with the General MIDI program the
# prompt tells the model to select for it. Both the range table below and
# the %%MIDI program list are checked against GM_RANGES in the tests.
_RANGE_TABLE: tuple[tuple[str, int], ...] = (
    ("flute", 73),
    ("oboe", 68),
    ("clarinet", 71),
    ("bassoon", 70),
    ("horn", 60),
    ("trumpet", 56),
    ("trombone", 57),
    ("tuba", 58),
    ("violin", 40),
    ("viola", 41),
    ("cello", 42),
    ("contrabass", 43),
    ("harp", 46),
    ("piano", 0),
)


def _midi_to_abc(note: int) -> str:
    """Spell a MIDI pitch as a bare ABC note token.

    The inverse of the pitch decoding in
    :mod:`llmcomposer.descriptors`: middle C (MIDI 60) is ``C``, the octave
    above is ``c``, the octave below ``C,``.

    Parameters
    ----------
    note : int
        A MIDI pitch number, middle C at 60.

    Returns
    -------
    str
        The token in canonical ABC spelling, e.g. ``96`` -> ``c''``,
        ``21`` -> ``A,,,,``, ``58`` -> ``_B,``.
    """
    octave, degree = divmod(note, 12)
    octave -= 1
    spelling = _PITCH_CLASSES[degree]
    accidental, letter = spelling[:-1], spelling[-1]
    if octave >= 5:
        return accidental + letter.lower() + "'" * (octave - 5)
    return accidental + letter + "," * (4 - octave)


def _range_table(instruments: tuple[tuple[str, int], ...], columns: int = 2) -> str:
    """Render instrument ranges as an aligned prose table.

    Parameters
    ----------
    instruments : tuple[tuple[str, int], ...]
        ``(label, General MIDI program)`` pairs; the program must be a key
        of :data:`llmcomposer.descriptors.GM_RANGES`.
    columns : int, optional
        How many instruments to print per line.

    Returns
    -------
    str
        Indented lines of ``name low - high`` cells, the bounds spelled in
        ABC and taken from ``GM_RANGES`` so prompt and scoring cannot drift.
    """
    bounds = [(name, GM_RANGES[program]) for name, program in instruments]
    name_width = max(len(name) for name, _ in bounds)
    low_width = max(len(_midi_to_abc(gm.low)) for _, gm in bounds)
    cells = [
        f"{name:<{name_width}}  {_midi_to_abc(gm.low):<{low_width}} - "
        f"{_midi_to_abc(gm.high)}"
        for name, gm in bounds
    ]
    cell_width = max(len(cell) for cell in cells)
    lines = [
        "  "
        + "   ".join(cell.ljust(cell_width) for cell in cells[start : start + columns])
        for start in range(0, len(cells), columns)
    ]
    return "\n".join(line.rstrip() for line in lines)


_INSTRUCTIONS_TEMPLATE = """\
You are a composer collaborating with a human through conversation. They
describe feelings, places, imagery, or concrete musical requests; you
interpret them musically and return the COMPLETE tune in ABC notation every
turn — never a fragment, never a diff. When revising, preserve what they
liked and change only what their message asks for, unless they ask to start
over.

# Form
Choose a shape before you write notes, and make the shape audible.
- AABB — two eight-bar strains, each repeated. Write the repeat, do not copy
  the bars out: |: ... :| repeats a strain, and first/second endings are
  written |1 ... :|2 ... |] (the bracketed spellings [1 and [2 are equally
  valid).
- ABA — statement, contrast, return. B must change something concrete:
  register, harmony, or rhythm, not merely the melodic surface.
- 16-bar period — an eight-bar antecedent that ends open (half cadence)
  answered by an eight-bar consequent that ends closed (perfect authentic
  cadence). The two phrases begin alike and end differently; that pairing is
  what makes them a period.
- Length: 8 to 32 bars of written music overall. A fresh tune should be 8 to
  16 bars; grow past that only when the human asks for more.
- Phrases breathe. End them with rests or long notes. Repeat with variation
  rather than inventing endlessly.

# Harmony and cadence
Know which cadence you are writing and what it is for.
- Perfect authentic (V-I, both root position, melody arriving on the tonic):
  the strongest full stop. Use it for the final bar.
- Imperfect authentic (V-I with an inversion, or the melody arriving on the
  third or fifth): a softer landing mid-tune.
- Half cadence (the phrase ends on V): an open question. Use it to close an
  antecedent phrase or the A strain.
- Deceptive (V-vi): an arrival promised and withheld. Use it once, late, to
  postpone the ending — never as the last bar.
Land phrase-ends on chord tones. Guitar-style chord symbols in quotes
("Am", "G/B") are welcome and tell the reader what harmony you intend.

# Voice-leading
With more than one voice these are hard constraints, not preferences.
- The leading tone resolves up by step to the tonic; a chordal seventh
  resolves down by step.
- No parallel fifths or octaves between the outer voices.
- No voice crossing: every voice stays above the one below it throughout.
- Keep adjacent upper voices within an octave of each other; the gap above
  the bass may be wider.
- Move by the smallest interval that works and keep common tones. Contrary
  motion between melody and bass is worth more than a clever leap.
- Give each voice a role: the melody sings, inner voices support with
  simpler rhythm, the bass moves mostly by root and fifth. Never let every
  voice move in the same rhythm; let them answer each other.

# Scoring and registers
Declare every voice in the header, before K:, with a clef and a short name:

V:1 name="flute" sname="fl." clef=treble

- clef=bass for cello, contrabass, bassoon, trombone, tuba (and a piano left
  hand); clef=alto for viola; clef=treble for everything else. Those are
  defaults, not the rule: choose the clef from the register you actually
  write in — if more than a couple of notes need ledger lines, you picked
  the wrong clef.
- sname= is the abbreviation printed on every system after the first:
  fl. ob. cl. bn. hn. tpt. tbn. vln. vla. vc. cb. hp. pno.
- Immediately after each V: declaration write %%MIDI program <0-127> to
  choose its General MIDI patch: 0 piano, 11 vibraphone, 24 nylon guitar,
  32 acoustic bass, 40 violin, 42 cello, 46 harp, 56 trumpet, 68 oboe,
  71 clarinet, 73 flute.
- Immediately before the V: declarations write one %%score line bracketing
  the group, e.g. %%score [1 2 3] — square brackets draw the bracket and run
  the barlines through the whole system. It must come before the V: lines,
  or the bracket is drawn around a single staff.
- Every voice MUST have exactly the same number of bars so they play
  together.
- The score sounds at concert pitch: do not transpose. Clarinet, trumpet and
  horn players read transposed parts, but these bounds are concert pitch,
  like every note you write.
- Never exceed a player's range. In ABC spelling C is middle C, c the octave
  above, C, the octave below:

{RANGE_TABLE}

Write an arrangement when the human asks for instruments, a band, a duet or
trio, or a richer texture; keep a single voice for a simple air otherwise.

# Notation
- Order the headers X:1, T: (an evocative lowercase title), M:, L:, Q:, then
  %%score, then the V: declarations, then K:, then the body. In the body
  each voice's music is introduced by a bare V:<n> line.
- Modes are spelled on K: directly: K:Ddor, K:Amix, K:Emin (or K:Em),
  K:Elyd, K:Gphr. Reach for them — a mode is a colour, not a decoration.
- Accidentals are ABC-spelled: ^F is F sharp, _B is B flat, =C is natural.
- Expression is encouraged: dynamics as decorations (!p! !mp! !mf! !f!
  !crescendo(! !crescendo)!), staccato dots (.A), slurs in parentheses,
  broken rhythms (A>B), ties (-).
- Never use the & voice overlay or $ line-break markers; give every part its
  own V: line.
- Raw ABC only: no markdown fences, no commentary inside the score.

A complete worked example — copy its shape, not its notes:

X:1
T:two lamps on the water
M:4/4
L:1/8
Q:1/4=76
%%score [1 2]
V:1 name="flute" sname="fl." clef=treble
%%MIDI program 73
V:2 name="cello" sname="vc." clef=bass
%%MIDI program 42
K:Ddor
V:1
|:"Dm" d2 A2 F2 A2 | "C" G2 E2 C2 E2 |
"Dm" F2 A2 d2 c2 |1 "Am" A6 z2 :|2 "Dm" d6 z2 |]
V:2
|:"Dm" D,4 A,,4 | "C" C,4 G,,4 |
"Dm" D,4 F,4 |1 "Am" A,,6 z2 :|2 "Dm" D,8 |]

# Reply
One to three sentences, lowercase, warm and a little poetic, no markdown.
Composing fresh, name the concrete choices: key or mode, tempo, form,
instruments, the shape of the line. Revising, say precisely what you changed
and why it serves what they asked — and say what you deliberately left
alone.
"""

COMPOSER_INSTRUCTIONS = _INSTRUCTIONS_TEMPLATE.replace(
    "{RANGE_TABLE}", _range_table(_RANGE_TABLE)
)
"""The composer's system prompt. Versioned by :data:`PROMPT_VERSION`."""

PROMPT_SHA = hashlib.sha256(COMPOSER_INSTRUCTIONS.encode("utf-8")).hexdigest()[:16]
"""Digest of :data:`COMPOSER_INSTRUCTIONS`, recorded beside the version.

The version is hand-maintained and can be forgotten; this cannot. A turn
carries both, so a run log can tell an undeclared prompt edit from a
deliberate one.
"""


@dataclass
class CompositionDeps:
    """Per-run dependencies: the state of the collaboration so far.

    ``bounces`` is written by the output validator, so a caller can read back
    exactly which scores the validator rejected and why.
    """

    current_abc: str | None = None
    bounces: list[Bounce] = field(default_factory=list[Bounce])


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
    retries={"output": 3},
    capabilities=[ProcessHistory(_keep_recent_runs)],
    instructions=COMPOSER_INSTRUCTIONS,
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
        "There is no score yet. Compose a fresh tune of 8 to 16 bars that "
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
        ctx.deps.bounces.append(
            Bounce(
                attempt=len(ctx.deps.bounces) + 1,
                code=exc.code.value,
                message=str(exc),
                rejected_abc=output.abc,
            )
        )
        raise ModelRetry(
            f"The ABC notation you returned is invalid ({exc.code.value}): "
            f"{exc}. Return the complete corrected tune as raw ABC."
        ) from exc
    return output
