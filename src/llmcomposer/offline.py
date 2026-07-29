"""An offline, deterministic composer for demos and tests without an API key.

Implemented as a :class:`pydantic_ai.models.function.FunctionModel`, so the
whole application stack — agent, validators, session, web app — runs
unchanged; only the model is swapped. Selected with ``LLMCOMPOSER_MODEL=offline``.
"""

from __future__ import annotations

import random
import re

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

_SCALES: dict[str, list[str]] = {
    "C": ["C", "D", "E", "F", "G", "A", "B", "c"],
    "G": ["G,", "A,", "B,", "C", "D", "E", "F", "G"],
    "Am": ["A,", "B,", "C", "D", "E", "F", "G", "A"],
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
            if headers and headers.group(1) in _SCALES:
                return headers.group(1)
    return None


def _compose_bars(scale: list[str], rng: random.Random, bars: int) -> list[str]:
    """Random-walk a melody over ``scale``; each bar is 8 eighth notes."""
    position = rng.randrange(2, 6)
    out: list[str] = []
    for bar in range(bars):
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


def _lower_octave(token: str) -> str:
    """Drop an ABC note token one octave (lowercase -> upper, upper -> ,)."""
    head = token.rstrip(",'")
    if head and head[-1].islower():
        return head[:-1] + head[-1].upper() + token[len(head) :].replace("'", "")
    return token + ","


def _bass_bars(scale: list[str], bars: int) -> list[str]:
    """Build root-and-fifth half notes below the melody's register."""
    root = _lower_octave(scale[0])
    fifth = _lower_octave(scale[4])
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


def _compose(prompt: str, key: str | None) -> tuple[str, str]:
    """Compose a reply and a complete multi-voice ABC arrangement."""
    mood_key, tempo, title = _pick_mood(prompt)
    key = key or mood_key
    rng = random.Random(hash(prompt) & 0xFFFFFFFF)
    scale = _SCALES[key]
    melody = _compose_bars(scale, rng, bars=8)

    voices = [("1", "flute", 73, _body_lines(melody))]
    instruments = ["flute"]
    if _wants_trio(prompt):
        voices.append(("2", "harp", 46, _body_lines(_harmony_bars(scale, 8))))
        instruments.append("harp")
    voices.append(
        (str(len(voices) + 1), "cello", 42, _body_lines(_bass_bars(scale, 8)))
    )
    instruments.append("cello")

    declarations = "".join(
        f'V:{vid} name="{name}"\n%%MIDI program {program}\n'
        for vid, name, program, _ in voices
    )
    body = "".join(f"V:{vid}\n{line}\n" for vid, _, _, line in voices)
    abc = f"X:1\nT:{title}\nM:4/4\nL:1/8\nQ:1/4={tempo}\n{declarations}K:{key}\n{body}"
    reply = (
        f"i heard '{prompt.strip()[:60]}' as {key} at {tempo} bpm — "
        f"{', '.join(instruments)} together, the melody wandering home to "
        "the tonic. tell me where to bend it."
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


def offline_model() -> FunctionModel:
    """Build the offline composer model.

    Returns
    -------
    FunctionModel
        A model producing valid, deterministic ABC tunes with no network.
    """
    return FunctionModel(_offline_composer, model_name="offline-composer")
