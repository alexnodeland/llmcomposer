"""An offline, deterministic composer for demos and tests without an API key.

Implemented as a :class:`pydantic_ai.models.function.FunctionModel`, so the
whole application stack — agent, validators, session, web app — runs
unchanged; only the model is swapped. Selected with ``LLMCOMPOSER_MODEL=offline``.

This is the study's null hypothesis: a keyword control, not a music-blind
one. It reads the prompt exactly twice — a small mood table picks the key,
tempo and title, and a second word list decides whether to add a third
voice — and it says so in its own reply. Nothing else about the text
reaches the music, and there is no musical understanding anywhere in it:
that is the floor a model has to beat.

The seed is a ``blake2b`` digest of the normalized prompt rather than
:func:`hash`, whose string hashing is salted per process — so the same words
give the same tune in every interpreter, forever.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)

_METER = "4/4"
_UNIT_LENGTH = "1/8"
_BARS = 8

# Every scale sits in the flute-safe octave (lowest note C4); lower voices
# are register-shifted copies, so no part is ever written under its range.
_BASE_SCALES: dict[str, list[str]] = {
    "C": ["C", "D", "E", "F", "G", "A", "B", "c"],
    "G": ["G", "A", "B", "c", "d", "e", "f", "g"],
    "Am": ["A", "B", "c", "d", "e", "f", "g", "a"],
    "Em": ["E", "F", "G", "A", "B", "c", "d", "e"],
    "Ddor": ["D", "E", "F", "G", "A", "B", "c", "d"],
}

_MOODS: list[tuple[tuple[str, ...], str, int, str]] = [
    (("rain", "sad", "melanchol", "grief", "night", "dark"), "Am", 66, "rainfall"),
    (("ocean", "sea", "dream", "drift", "calm", "still"), "Ddor", 72, "slack tide"),
    (("bright", "dawn", "morning", "hope", "joy", "happy"), "G", 100, "first light"),
    (("forest", "garden", "moss", "green", "grow"), "Em", 84, "understory"),
]

_SCORE_TAG = re.compile(r"<current_score>\n(.*?)\n</current_score>", re.DOTALL)
_NOTE_TOKEN = re.compile(r"^([_^=]*)([A-Ga-g])([,']*)$")


@dataclass(frozen=True)
class _Voice:
    """One staff of the baseline arrangement."""

    name: str
    sname: str
    clef: str
    program: int
    octaves: int


_FLUTE = _Voice(name="flute", sname="fl.", clef="treble", program=73, octaves=0)
# The harp's ostinato is written an octave down (D3-A3), so it reads on a
# bass staff; on a treble one every note of it would sit on ledger lines.
# The offsets are fixed rather than re-anchored per key: in the higher keys
# (Am) the harp drifts a ledger line or two above the bass staff, but every
# per-key re-anchoring either collides the harp with the cello (introducing
# voice crossings) or pushes the cello below its playable C2 — a few ledger
# lines is the cheapest of the three defects.
_HARP = _Voice(name="harp", sname="hp.", clef="bass", program=46, octaves=-1)
_CELLO = _Voice(name="cello", sname="vc.", clef="bass", program=42, octaves=-2)


def _seed(prompt: str) -> int:
    """Derive a process-stable RNG seed from the normalized prompt."""
    normalized = " ".join(prompt.lower().split()).encode()
    return int.from_bytes(hashlib.blake2b(normalized, digest_size=4).digest(), "big")


def _last_user_prompt(messages: list[ModelMessage]) -> str:
    """Return the most recent user prompt text in the conversation."""
    for message in reversed(messages):
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                return part.content
    return ""


def _pick_mood(prompt: str) -> tuple[str, int, str]:
    """Map feeling-words in the prompt to a key, tempo, and title."""
    lowered = prompt.lower()
    for words, key, tempo, title in _MOODS:
        if any(word in lowered for word in words):
            return key, tempo, title
    return "Ddor", 76, "small hours"


def _existing_key(messages: list[ModelMessage]) -> str | None:
    """Recover the working score's key from the injected instructions."""
    for message in reversed(messages):
        instructions = getattr(message, "instructions", None)
        if not instructions:
            continue
        match = _SCORE_TAG.search(instructions)
        if match:
            headers = re.search(r"^K:\s*(\S+)", match.group(1), re.MULTILINE)
            if headers and headers.group(1) in _BASE_SCALES:
                return headers.group(1)
    return None


def _transpose(token: str, octaves: int) -> str:
    """Shift a bare ABC note token by whole octaves.

    Parameters
    ----------
    token : str
        A note token with no duration, e.g. ``"C"``, ``"^f"``, ``"A,"``.
    octaves : int
        How many octaves to move; negative goes down.

    Returns
    -------
    str
        The transposed token in canonical ABC octave spelling.
    """
    match = _NOTE_TOKEN.match(token)
    if match is None:
        raise ValueError(f"'{token}' is not a bare ABC note token")
    accidental, letter, marks = match.groups()
    octave = (4 if letter.isupper() else 5) + marks.count("'") - marks.count(",")
    octave += octaves
    if octave >= 5:
        return accidental + letter.lower() + "'" * (octave - 5)
    return accidental + letter.upper() + "," * (4 - octave)


def _register(scale: list[str], octaves: int) -> list[str]:
    """Return ``scale`` shifted into another voice's register."""
    return [_transpose(token, octaves) for token in scale]


def _compose_bars(scale: list[str], rng: random.Random, bars: int) -> list[str]:
    """Random-walk a melody over ``scale``; each bar is 8 eighth notes."""
    position = rng.randrange(2, 6)
    out: list[str] = []
    for _ in range(bars):
        beats: list[str] = []
        for _ in range(4):
            position = max(0, min(7, position + rng.choice([-2, -1, -1, 1, 1, 2])))
            first = scale[position]
            if rng.random() < 0.3:
                beats.append(f"{first}2")
            else:
                position = max(0, min(7, position + rng.choice([-1, 1])))
                beats.append(f"{first}{scale[position]}")
        out.append(" ".join(beats))
    out[-1] = f"{scale[4]}4 {scale[0]}4"
    return out


def _bass_bars(scale: list[str], bars: int) -> list[str]:
    """Build root-and-fifth half notes in the bass voice's own register."""
    root, fifth = scale[0], scale[4]
    line = [f"{root}4 {fifth}4" for _ in range(bars)]
    line[-1] = f"{root}8"
    return line


def _harmony_bars(scale: list[str], bars: int) -> list[str]:
    """Build a gentle tonic-triad arpeggio pattern in quarter notes."""
    a, b, c = scale[0], scale[2], scale[4]
    line = [f"{a}2 {b}2 {c}2 {b}2" for _ in range(bars)]
    line[-1] = f"{a}8"
    return line


def _wants_trio(prompt: str) -> bool:
    """Return whether the prompt asks for a fuller arrangement."""
    lowered = prompt.lower()
    words = ("trio", "arrange", "band", "ensemble", "instrument", "together")
    return any(word in lowered for word in words)


def _body_lines(bars: list[str], closing: str = " |]") -> str:
    """Lay the bars out four to a line."""
    first = " | ".join(bars[:4]) + " |"
    second = " | ".join(bars[4:]) + closing
    return f"{first}\n{second}"


def _arrangement(prompt: str, key: str, rng: random.Random) -> list[tuple[_Voice, str]]:
    """Choose the voices and write each one's bars in its own register."""
    base = _BASE_SCALES[key]
    parts: list[tuple[_Voice, list[str]]] = [
        (_FLUTE, _compose_bars(_register(base, _FLUTE.octaves), rng, _BARS))
    ]
    if _wants_trio(prompt):
        parts.append((_HARP, _harmony_bars(_register(base, _HARP.octaves), _BARS)))
    parts.append((_CELLO, _bass_bars(_register(base, _CELLO.octaves), _BARS)))
    return [(voice, _body_lines(bars)) for voice, bars in parts]


def _compose(prompt: str, key: str | None) -> tuple[str, str]:
    """Compose a reply and a complete multi-voice ABC arrangement."""
    mood_key, tempo, title = _pick_mood(prompt)
    key = key or mood_key
    parts = _arrangement(prompt, key, random.Random(_seed(prompt)))

    declarations = "".join(
        f'V:{n} name="{voice.name}" sname="{voice.sname}" clef={voice.clef}\n'
        f"%%MIDI program {voice.program}\n"
        for n, (voice, _) in enumerate(parts, start=1)
    )
    staves = " ".join(str(n) for n in range(1, len(parts) + 1))
    body = "".join(f"V:{n}\n{lines}\n" for n, (_, lines) in enumerate(parts, start=1))
    # %%score must precede the V: declarations or the bracket is engraved
    # around the first staff alone instead of around the system.
    abc = (
        f"X:1\nT:{title}\nM:{_METER}\nL:{_UNIT_LENGTH}\nQ:1/4={tempo}\n"
        f"%%score [{staves}]\n{declarations}K:{key}\n{body}"
    )
    reply = (
        f"baseline: {key} at {tempo} bpm, chosen by keyword match from a "
        "small mood table — no musical understanding, just the floor to beat."
    )
    return reply, abc


def _offline_composer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """FunctionModel implementation backing :func:`offline_model`."""
    prompt = _last_user_prompt(messages)
    reply, abc = _compose(prompt, _existing_key(messages))
    if not info.output_tools:
        return ModelResponse(parts=[])
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name=info.output_tools[0].name,
                args={"reply": reply, "abc": abc},
            )
        ]
    )


async def _offline_composer_stream(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[DeltaToolCalls]:
    """Stream the same composition as chunked tool-call deltas."""
    prompt = _last_user_prompt(messages)
    reply, abc = _compose(prompt, _existing_key(messages))
    if not info.output_tools:
        return
    args = json.dumps({"reply": reply, "abc": abc})
    yield {0: DeltaToolCall(name=info.output_tools[0].name)}
    for start in range(0, len(args), 48):
        yield {0: DeltaToolCall(json_args=args[start : start + 48])}


def offline_model() -> FunctionModel:
    """Build the offline composer model.

    Returns
    -------
    FunctionModel
        A model producing valid, deterministic ABC tunes with no network.
        Supports both plain and streamed runs.
    """
    return FunctionModel(
        _offline_composer,
        stream_function=_offline_composer_stream,
        model_name="offline-composer",
    )
