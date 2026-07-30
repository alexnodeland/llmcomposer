"""Validation and light parsing for ABC music notation.

The composer agent returns whole tunes in ABC notation. These helpers do a
real parse of the tune body — tokenizing notes, rests, chords, and tuplets,
and checking bar durations against the meter — so the agent's output
validator can kick malformed scores back to the model (via ``ModelRetry``)
with a precise, actionable reason.

Why not an existing library: pyabc2 and music21 (checked 2026-07) are both
*lenient* parsers — they silently drop invalid tokens (e.g. ``F#`` for
``^F``) and never check bar durations against the meter. Error recovery is
the wrong behavior here; the retry loop needs strict rejection with a
message the model can act on.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from fractions import Fraction

REQUIRED_HEADERS = ("X", "M", "L", "K")

_HEADER_LINE = re.compile(r"^[A-Za-z]:")
_METER = re.compile(r"^(\d+)/(\d+)$")
_UNIT_LENGTH = re.compile(r"^(\d+)/(\d+)$")

# Constructs that carry no duration: chord symbols / annotations, dynamics
# and decorations (!p!, staccato dots, letter shortcuts), grace notes,
# invisible spacers, broken-rhythm markers, ties, and slurs.
_NEUTRAL = re.compile(
    r'"[^"]*"|![^!]*!|\{[^}]*\}|y\d*|[<>\-\\)]|\((?!\d)|[.~HLMOPSTuv]'
)

_BARLINE = re.compile(r"\[\||\|\]|:\|+|\|+:|::|\|")
_VOLTA = re.compile(r"^\s*\[?\d[\s,-]*")
_REPEAT_MARKS = re.compile(r":\||\|:|\[\d|\|\d")

_NOTE = re.compile(r"[_^=]{0,2}[A-Ga-g][,']*(\d*(?:/\d*)*)")
_REST = re.compile(r"[zx](\d*(?:/\d*)*)")
_BIG_REST = re.compile(r"[ZX](\d*)")
_CHORD = re.compile(r"\[([^\]]*)\](\d*(?:/\d*)*)")
_TUPLET = re.compile(r"\((\d)(?::(\d?))?(?::(\d?))?")
_INLINE_FIELD = re.compile(r"\[[A-Za-z]:[^\]]*\]")

_TUPLET_TIME = {2: 3, 3: 2, 4: 3, 6: 2, 8: 3}


class ABCValidationError(ValueError):
    """Raised when a string is not a playable ABC tune."""


def parse_headers(abc: str) -> dict[str, str]:
    """Parse ABC header fields into a mapping.

    Parameters
    ----------
    abc : str
        The ABC source of a single tune.

    Returns
    -------
    dict[str, str]
        Header letters mapped to their (last seen) values.
    """
    headers: dict[str, str] = {}
    for line in abc.splitlines():
        stripped = line.strip()
        if _HEADER_LINE.match(stripped):
            headers[stripped[0]] = stripped[2:].strip()
    return headers


def music_lines(abc: str) -> list[str]:
    """Return the tune body: lines that are not header fields or comments.

    Parameters
    ----------
    abc : str
        The ABC source of a single tune.

    Returns
    -------
    list[str]
        The lines carrying the notes themselves.
    """
    body: list[str] = []
    for line in abc.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if _HEADER_LINE.match(stripped):
            continue
        body.append(stripped)
    return body


_DURATION = re.compile(r"^(\d+)?((?:/\d*)*)$")


def _duration(text: str) -> Fraction:
    """Convert an ABC duration suffix (``''``, ``2``, ``3/2``, ``/``, ``//``)."""
    match = _DURATION.match(text)
    if match is None:
        raise ABCValidationError(f"'{text}' is not a valid note duration")
    value = Fraction(int(match.group(1))) if match.group(1) else Fraction(1)
    for denominator in re.findall(r"/(\d*)", match.group(2) or ""):
        value /= int(denominator) if denominator else 2
    return value


def _meter_units(headers: dict[str, str]) -> Fraction | None:
    """Bar length as a fraction of a whole note, or ``None`` if free meter."""
    meter = headers.get("M", "").strip()
    if meter in {"C", "c"}:
        return Fraction(4, 4)
    if meter in {"C|", "c|"}:
        return Fraction(2, 2)
    match = _METER.match(meter)
    if match:
        return Fraction(int(match.group(1)), int(match.group(2)))
    if meter.lower() == "none":
        return None
    raise ABCValidationError(f"the M: header '{meter}' is not a valid meter")


def _unit_length(headers: dict[str, str]) -> Fraction:
    """Return the L: unit note length as a fraction of a whole note."""
    unit = headers.get("L", "").strip()
    match = _UNIT_LENGTH.match(unit)
    if not match:
        raise ABCValidationError(
            f"the L: header '{unit}' is not a valid unit note length"
        )
    return Fraction(int(match.group(1)), int(match.group(2)))


class _BarParser:
    """Tokenizes one bar and accumulates its duration in unit-note lengths."""

    def __init__(self, bar: str, full_bar_units: Fraction) -> None:
        self.text = _NEUTRAL.sub(" ", _VOLTA.sub("", bar))
        self.total = Fraction(0)
        self.verifiable = True
        self.full_bar_units = full_bar_units
        self.tuplet_factor = Fraction(1)
        self.tuplet_remaining = 0

    def _add(self, duration: Fraction) -> None:
        if self.tuplet_remaining > 0:
            duration *= self.tuplet_factor
            self.tuplet_remaining -= 1
        self.total += duration

    def _match_token(self, pos: int) -> int | None:
        handlers: tuple[
            tuple[re.Pattern[str], Callable[[re.Match[str]], None]], ...
        ] = (
            (_INLINE_FIELD, self._on_inline_field),
            (_TUPLET, self._on_tuplet),
            (_CHORD, self._on_chord),
            (_NOTE, lambda m: self._add(_duration(m.group(1)))),
            (_REST, lambda m: self._add(_duration(m.group(1)))),
            (_BIG_REST, self._on_big_rest),
        )
        for regex, handler in handlers:
            match = regex.match(self.text, pos)
            if match:
                handler(match)
                return match.end()
        return None

    def _on_inline_field(self, match: re.Match[str]) -> None:
        self.verifiable = False

    def _on_tuplet(self, match: re.Match[str]) -> None:
        notes = int(match.group(1))
        in_time_of = match.group(2)
        time = int(in_time_of) if in_time_of else _TUPLET_TIME.get(notes)
        if time is None:
            self.verifiable = False
            return
        self.tuplet_factor = Fraction(time, notes)
        self.tuplet_remaining = int(match.group(3) or notes)

    def _on_chord(self, match: re.Match[str]) -> None:
        inner = [_duration(note.group(1)) for note in _NOTE.finditer(match.group(1))]
        if not inner:
            raise ABCValidationError(
                f"'[{match.group(1)}]' is not a valid chord: no notes inside"
            )
        self._add(max(inner) * _duration(match.group(2)))

    def _on_big_rest(self, match: re.Match[str]) -> None:
        # Z counts whole bars, so it can only ever fill bars exactly.
        bars = int(match.group(1) or 1)
        self.total += bars * self.full_bar_units

    def parse(self) -> tuple[Fraction, bool]:
        """Return the bar's duration (in unit-note lengths) and verifiability."""
        pos = 0
        while pos < len(self.text):
            if self.text[pos].isspace():
                pos += 1
                continue
            advanced = self._match_token(pos)
            if advanced is None:
                snippet = self.text[pos : pos + 8].strip()
                raise ABCValidationError(
                    f"unparseable ABC at '{snippet}' — not a note, rest, "
                    "chord, tuplet, or barline"
                )
            pos = advanced
        return self.total, self.verifiable


def voice_bars(abc: str) -> dict[str, list[str]]:
    """Split the tune body into bars, grouped by voice.

    Single-voice tunes come back under the key ``"1"``. ``V:`` lines in the
    body switch the current voice; ``V:`` lines before the body only declare
    voices and their instruments.

    Parameters
    ----------
    abc : str
        The ABC source of a single tune.

    Returns
    -------
    dict[str, list[str]]
        Voice id mapped to that voice's bars, in order.
    """
    voices: dict[str, list[str]] = {}
    current = "1"
    in_body = False
    for line in abc.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if stripped.startswith("K:"):
            in_body = True
            continue
        if stripped.startswith("V:"):
            current = stripped[2:].split()[0] if stripped[2:].split() else "1"
            continue
        if _HEADER_LINE.match(stripped):
            continue
        if in_body:
            bars = voices.setdefault(current, [])
            bars.extend(seg for seg in _BARLINE.split(stripped) if seg.strip())
    return voices


def _check_voice(
    voice: str,
    bars: list[str],
    full_bar: Fraction,
    headers: dict[str, str],
    has_repeats: bool,
    multi_voice: bool,
) -> None:
    """Verify one voice's bars against the meter."""
    label = f"voice {voice}, " if multi_voice else ""
    for index, bar in enumerate(bars):
        duration, verifiable = _BarParser(bar, full_bar).parse()
        if not verifiable:
            continue
        is_edge = index in (0, len(bars) - 1)
        if duration > full_bar or (
            duration != full_bar and not is_edge and not has_repeats
        ):
            unit = _unit_length(headers)
            raise ABCValidationError(
                f"{label}bar {index + 1} ('{bar.strip()}') lasts "
                f"{duration * unit} of a whole note but the meter "
                f"{headers['M']} needs {_meter_units(headers)} per bar"
            )


def _check_bar_durations(abc: str, headers: dict[str, str]) -> None:
    """Verify every voice's bar durations, and that voices stay aligned."""
    meter = _meter_units(headers)
    if meter is None:
        return
    full_bar = meter / _unit_length(headers)
    has_repeats = bool(_REPEAT_MARKS.search(abc))
    voices = voice_bars(abc)
    multi_voice = len(voices) > 1
    for voice, bars in voices.items():
        _check_voice(voice, bars, full_bar, headers, has_repeats, multi_voice)
    if multi_voice:
        counts = {voice: len(bars) for voice, bars in voices.items()}
        if len(set(counts.values())) > 1:
            described = ", ".join(f"voice {v}: {n} bars" for v, n in counts.items())
            raise ABCValidationError(
                f"the voices are different lengths ({described}); every voice "
                "must have the same number of bars so they play together"
            )


def validate_abc(abc: str) -> None:
    """Check that ``abc`` is a single, parseable, playable ABC tune.

    Parameters
    ----------
    abc : str
        Candidate ABC source.

    Raises
    ------
    ABCValidationError
        With a human-readable reason when the tune is malformed. The message
        is written to be shown back to the model so it can self-correct.
    """
    stripped = abc.strip()
    if not stripped:
        raise ABCValidationError("the score is empty")
    if "```" in stripped:
        raise ABCValidationError(
            "the score contains markdown code fences; return raw ABC only"
        )

    headers = parse_headers(stripped)
    missing = [h for h in REQUIRED_HEADERS if h not in headers]
    if missing:
        fields = ", ".join(f"{h}:" for h in missing)
        raise ABCValidationError(f"missing required header field(s): {fields}")

    first_line = stripped.splitlines()[0].strip()
    if not first_line.startswith("X:"):
        raise ABCValidationError("the first line must be the X: reference number")

    body = music_lines(stripped)
    if not body:
        raise ABCValidationError("the score has headers but no music lines")
    if not any("|" in line for line in body):
        raise ABCValidationError("the music lines contain no bar lines ('|')")

    _check_bar_durations(stripped, headers)
