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

The instrument is meant to be trustworthy in both directions. It must not
accept a tune whose bars do not add up (there are no blanket exemptions —
a repeat sign never buys a bar a free pass, and a pickup is only a pickup
if the final bar completes it), and it must not reject ABC that any tune
book would print (``[|:``, ``:|]``, ``::``, multi-measure rests, inline
``[M:]``/``[L:]`` changes, and irregular tuplets all parse).
Every failure carries an :class:`ABCErrorCode` so bounces can be counted by
kind rather than by regex over prose.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

REQUIRED_HEADERS = ("X", "M", "L", "K")


class ABCErrorCode(StrEnum):
    """Machine-readable classification of an ABC validation failure."""

    EMPTY = "empty"
    FENCED = "fenced"
    MISSING_HEADER = "missing_header"
    BAD_FIRST_LINE = "bad_first_line"
    NO_BODY = "no_body"
    NO_BARLINES = "no_barlines"
    BAD_METER = "bad_meter"
    BAD_UNIT_LENGTH = "bad_unit_length"
    BAD_DURATION = "bad_duration"
    BAD_KEY = "bad_key"
    BAD_TEMPO = "bad_tempo"
    MULTIPLE_TUNES = "multiple_tunes"
    BAD_MIDI_PROGRAM = "bad_midi_program"
    EMPTY_CHORD = "empty_chord"
    BAR_LENGTH = "bar_length"
    VOICE_MISALIGNED = "voice_misaligned"
    UNPARSEABLE = "unparseable"


class ABCValidationError(ValueError):
    """Raised when a string is not a playable ABC tune.

    Parameters
    ----------
    message : str
        Human-readable reason. Written to be shown back to the model, so it
        names the offending bar and says what would fix it.
    code : ABCErrorCode, optional
        Machine-readable classification, for counting bounces by kind.
        Defaults to :attr:`ABCErrorCode.UNPARSEABLE`.
    """

    code: ABCErrorCode

    def __init__(
        self, message: str, code: ABCErrorCode = ABCErrorCode.UNPARSEABLE
    ) -> None:
        super().__init__(message)
        self.code = code


_HEADER_LINE = re.compile(r"^[A-Za-z]:")
_METER = re.compile(r"^\(?(\d+(?:\s*\+\s*\d+)*)\)?/(\d+)$")
_UNIT_LENGTH = re.compile(r"^(\d+)/(\d+)$")

# A key is a letter with an optional accidental and an optional mode, and may
# be followed by extra fields (clef=, exp, transpose=…). ``none``/``HP``/``Hp``
# are the bagpipe and no-key spellings.
_MODES = "maj|ion|min|aeo|mix|dor|phr|lyd|loc|m"
_KEY = re.compile(
    rf"^(?:none|HP|Hp|[A-Ga-g][#b]?(?:\s*(?:{_MODES})[a-z]*)?)(?:\s+.+)?$",
    re.IGNORECASE,
)
_TEMPO_TEXT = re.compile(r'"[^"]*"')
_TEMPO_BEATS = re.compile(r"^\d+/\d+(?:\s+\d+/\d+)*\s*=\s*\d+(?:\.\d+)?$")
_MIDI_PROGRAM = re.compile(r"^\s*%%\s*MIDI\s+program\b(.*)$", re.MULTILINE)

# Constructs that carry no duration: chord symbols / annotations, dynamics
# and decorations (!p!, +p+, staccato dots, letter shortcuts), grace notes,
# invisible spacers, broken-rhythm markers, ties, slurs, and the ``$``
# linebreak marker (ABC 2.1 ``I:linebreak $``).
_NEUTRAL = re.compile(
    r'"[^"]*"|![^!]*!|\+[^+]*\+|\{[^}]*\}|y\d*|[<>\-\\)$]|\((?!\d)|[.~HLMOPSTuv]'
)

# The full ABC 2.1 barline set, longest alternative first so that ``[|:`` and
# ``:|]`` tokenize whole instead of orphaning a colon or a bracket.
_BARLINE = re.compile(r"(:\|+:|::|:\|+\]?|\[\|:|\[\||\|+:|\|+\]|\|+)")
_VOLTA = re.compile(r"^\s*\[?\d[\d,\-]*\s*")

_NOTE = re.compile(r"[_^=]{0,2}[A-Ga-g][,']*(\d*(?:/\d*)*)")
_REST = re.compile(r"[zx](\d*(?:/\d*)*)")
_BIG_REST = re.compile(r"[ZX](\d*)")
_CHORD = re.compile(r"\[([^\]]*)\](\d*(?:/\d*)*)")
_TUPLET = re.compile(r"\((\d)(?::(\d?))?(?::(\d?))?")
_INLINE_FIELD = re.compile(r"\[([A-Za-z]):([^\]]*)\]")

# ABC 2.2: (2 (3 (4 (6 (8 have fixed ratios; (5 (7 (9 depend on the meter.
_TUPLET_TIME = {2: 3, 3: 2, 4: 3, 6: 2, 8: 3}

_DURATION = re.compile(r"^(\d+)?((?:/\d*)*)$")


@dataclass(frozen=True)
class _Meter:
    """A meter: its bar length as a fraction of a whole note, plus its face."""

    bar: Fraction | None
    compound: bool
    text: str


@dataclass
class _State:
    """The meter and unit note length currently in force within a voice."""

    meter: _Meter
    unit: Fraction


@dataclass
class _Segment:
    """One bar of music together with the barlines that bracket it."""

    text: str
    left: str = ""
    right: str = ""


@dataclass
class _BarResult:
    """What one bar turned out to hold."""

    duration: Fraction
    bars: int = 0


@dataclass(frozen=True)
class _Parsed:
    """One bar's parse, together with the meter that was in force for it."""

    result: _BarResult
    meter: _Meter


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


def _strip_comment(raw: str) -> str:
    """Drop a whole-line comment or directive, and any trailing ``%`` remark."""
    line = raw.strip()
    if line.startswith("%"):
        return ""
    index = line.find("%")
    return line[:index].strip() if index >= 0 else line


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
        stripped = _strip_comment(line)
        if not stripped or _HEADER_LINE.match(stripped):
            continue
        body.append(stripped)
    return body


def _duration(text: str, token: str = "") -> Fraction:
    """Convert an ABC duration suffix (``''``, ``2``, ``3/2``, ``/``, ``//``)."""
    match = _DURATION.match(text)
    if match is None:
        raise ABCValidationError(
            f"'{token or text}' is not a valid note duration",
            ABCErrorCode.BAD_DURATION,
        )
    value = Fraction(int(match.group(1))) if match.group(1) else Fraction(1)
    for denominator in re.findall(r"/(\d*)", match.group(2) or ""):
        divisor = int(denominator) if denominator else 2
        if divisor == 0:
            value = Fraction(0)
            break
        value /= divisor
    if value == 0:
        raise ABCValidationError(
            f"'{token or text}' has a length of zero; every note, rest and "
            "chord needs a duration of at least one unit (write C or C1, not C0)",
            ABCErrorCode.BAD_DURATION,
        )
    return value


def _parse_meter(text: str) -> _Meter:
    """Read an ``M:`` value into a bar length plus a simple/compound flag."""
    meter = text.strip()
    if meter in {"C", "c"}:
        return _Meter(Fraction(4, 4), False, meter)
    if meter in {"C|", "c|"}:
        return _Meter(Fraction(2, 2), False, meter)
    if meter.lower() == "none":
        return _Meter(None, False, meter)
    match = _METER.match(meter)
    if match is None:
        raise ABCValidationError(
            f"the M: header '{meter}' is not a meter i can read; "
            "write it as 4/4, 6/8, 2/2, C, C| or none",
            ABCErrorCode.BAD_METER,
        )
    beats = sum(int(part) for part in match.group(1).replace(" ", "").split("+"))
    unit = int(match.group(2))
    if beats == 0:
        raise ABCValidationError(
            f"the M: header '{meter}' has no beats in it",
            ABCErrorCode.BAD_METER,
        )
    compound = unit in {8, 16} and beats > 3 and beats % 3 == 0
    return _Meter(Fraction(beats, unit), compound, meter)


def _unit_length(text: str) -> Fraction:
    """Read an ``L:`` value as a fraction of a whole note."""
    unit = text.strip()
    match = _UNIT_LENGTH.match(unit)
    if not match or int(match.group(1)) == 0 or int(match.group(2)) == 0:
        raise ABCValidationError(
            f"the L: header '{unit}' is not a unit note length; "
            "write it as 1/8, 1/16 or 1/4",
            ABCErrorCode.BAD_UNIT_LENGTH,
        )
    return Fraction(int(match.group(1)), int(match.group(2)))


def _tuplet_time(notes: int, compound: bool) -> int:
    """Return the ``q`` a bare ``(n`` tuplet stands for, per ABC 2.2."""
    fixed = _TUPLET_TIME.get(notes)
    if fixed is not None:
        return fixed
    return 3 if compound else 2


class _BarParser:
    """Tokenizes one bar and accumulates its duration as part of a whole note.

    The parser reads and writes ``state``: an inline ``[M:…]`` or ``[L:…]``
    field changes the meter or unit note length for the rest of the voice.
    Every other inline field is duration-neutral and simply skipped, so the
    bar stays verifiable.
    """

    def __init__(self, bar: str, state: _State) -> None:
        self.text = _VOLTA.sub("", bar)
        self.state = state
        self.bar_meter = state.meter
        self.total = Fraction(0)
        self.bars = 0
        self.tuplet_factor = Fraction(1)
        self.tuplet_remaining = 0

    def _add(self, duration: Fraction) -> None:
        if self.tuplet_remaining > 0:
            duration *= self.tuplet_factor
            self.tuplet_remaining -= 1
        self.total += duration * self.state.unit

    def _on_inline_field(self, match: re.Match[str]) -> None:
        letter = match.group(1).upper()
        value = match.group(2)
        if letter == "M":
            self.state.meter = _parse_meter(value)
            if self.total == 0:
                self.bar_meter = self.state.meter
        elif letter == "L":
            self.state.unit = _unit_length(value)

    def _on_tuplet(self, match: re.Match[str]) -> None:
        notes = int(match.group(1))
        in_time_of = match.group(2)
        time = (
            int(in_time_of)
            if in_time_of
            else _tuplet_time(notes, self.state.meter.compound)
        )
        self.tuplet_factor = Fraction(time, notes)
        self.tuplet_remaining = int(match.group(3) or notes)

    def _on_chord(self, match: re.Match[str]) -> None:
        inner = [_duration(note.group(1)) for note in _NOTE.finditer(match.group(1))]
        if not inner:
            raise ABCValidationError(
                f"'[{match.group(1)}]' is not a valid chord: no notes inside",
                ABCErrorCode.EMPTY_CHORD,
            )
        self._add(max(inner) * _duration(match.group(2), match.group(0)))

    def _on_big_rest(self, match: re.Match[str]) -> None:
        # Z/X count whole bars, so they fill exactly the bars they claim.
        count = int(match.group(1) or 1)
        self.bars += count
        bar = self.state.meter.bar
        if bar is not None:
            self.total += count * bar

    def _handlers(
        self,
    ) -> tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], None]], ...]:
        return (
            (_INLINE_FIELD, self._on_inline_field),
            (_NEUTRAL, lambda _: None),
            (_TUPLET, self._on_tuplet),
            (_CHORD, self._on_chord),
            (_NOTE, lambda m: self._add(_duration(m.group(1), m.group(0)))),
            (_REST, lambda m: self._add(_duration(m.group(1), m.group(0)))),
            (_BIG_REST, self._on_big_rest),
        )

    def _match_token(self, pos: int) -> int | None:
        for regex, handler in self._handlers():
            match = regex.match(self.text, pos)
            if match:
                handler(match)
                return match.end()
        return None

    def parse(self) -> _BarResult:
        """Return the bar's duration (as part of a whole note) and bars filled."""
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
                    "chord, tuplet, or barline",
                    ABCErrorCode.UNPARSEABLE,
                )
            pos = advanced
        return _BarResult(self.total, self.bars)


def _split_bars(text: str) -> list[_Segment]:
    """Split a run of music into bars, remembering each bar's own barlines."""
    parts = _BARLINE.split(text)
    segments: list[_Segment] = []
    pending = ""
    open_bar: int | None = None
    for index, part in enumerate(parts):
        if index % 2 == 0:
            if part.strip():
                segments.append(_Segment(text=part, left=pending))
                open_bar = len(segments) - 1
                pending = ""
            continue
        if open_bar is not None:
            segments[open_bar].right = part
            open_bar = None
        pending = part
    return segments


def _voice_text(abc: str) -> dict[str, str]:
    """Join each voice's body lines into one run of music."""
    voices: dict[str, str] = {}
    current = "1"
    in_body = False
    for raw in abc.splitlines():
        line = _strip_comment(raw)
        if not line:
            continue
        if line.startswith("K:"):
            if not in_body:
                in_body, current = True, "1"
            continue
        if line.startswith("V:"):
            fields = line[2:].split()
            current = fields[0] if fields else "1"
            continue
        if _HEADER_LINE.match(line):
            continue
        if in_body:
            voices[current] = f"{voices.get(current, '')} {line}"
    return voices


def voice_bars(abc: str) -> dict[str, list[str]]:
    """Split the tune body into bars, grouped by voice.

    Single-voice tunes come back under the key ``"1"``. ``V:`` lines in the
    body switch the current voice; ``V:`` lines before the body only declare
    voices and their instruments. A line that does not end on a barline runs
    on into the next one, so a bar broken across lines stays one bar.

    Parameters
    ----------
    abc : str
        The ABC source of a single tune.

    Returns
    -------
    dict[str, list[str]]
        Voice id mapped to that voice's bars, in order.
    """
    return {
        voice: [segment.text.strip() for segment in _split_bars(text)]
        for voice, text in _voice_text(abc).items()
    }


def _opens_section(barline: str) -> bool:
    """Report whether a barline opens a repeat (``|:``, ``[|:``, ``::``)."""
    return "|:" in barline or "::" in barline


def _closes_section(barline: str) -> bool:
    """Report whether a barline closes a repeat (``:|``, ``:|]``, ``::``)."""
    return ":|" in barline or "::" in barline


def _completes_a_bar(parsed: list[_Parsed]) -> bool:
    """Report whether a voice's first and last bar are a true anacrusis pair.

    The pickup is borrowed from the final bar, so the two together must
    fill exactly one bar. A full first bar therefore demands a full last
    one, and vice versa — the sum is the whole test.
    """
    if len(parsed) < 2:
        return False
    total = parsed[0].result.duration + parsed[-1].result.duration
    return any(
        edge.meter.bar is not None and total == edge.meter.bar
        for edge in (parsed[0], parsed[-1])
    )


def _is_section_edge(
    index: int, segments: list[_Segment], voice: list[_Parsed]
) -> bool:
    """Report whether a short bar would be musically legitimate here.

    A bar may be short at the edge of a section: a bar that runs into a
    repeat end, a bar that follows a repeat start, or a bar that opens a
    volta. The first and last bar of a voice are exempt only as a *pair* —
    when they add up to one bar between them, which is what makes a pickup
    a pickup. Everywhere else the meter is the meter.
    """
    if index in (0, len(segments) - 1) and _completes_a_bar(voice):
        return True
    segment = segments[index]
    if _closes_section(segment.right) or _opens_section(segment.left):
        return True
    return bool(_VOLTA.match(segment.text))


def _check_bar(
    label: str,
    index: int,
    segments: list[_Segment],
    voice: list[_Parsed],
) -> None:
    """Compare one bar against the meter in force for it."""
    result, meter = voice[index].result, voice[index].meter
    full = meter.bar
    if full is None:
        return
    expected = full * result.bars if result.bars else full
    if result.duration == expected:
        return
    where = f"{label}bar {index + 1} ('{segments[index].text.strip()}')"
    if result.duration > expected:
        raise ABCValidationError(
            f"{where} holds {result.duration} of a whole note, more than the "
            f"meter {meter.text} allows ({expected} per bar); shorten the notes "
            "or move the barline",
            ABCErrorCode.BAR_LENGTH,
        )
    if _is_section_edge(index, segments, voice):
        return
    raise ABCValidationError(
        f"{where} holds {result.duration} of a whole note but the meter "
        f"{meter.text} needs {expected}; a bar may be short only at a repeat "
        "or volta edge, or as a pickup whose final bar completes it",
        ABCErrorCode.BAR_LENGTH,
    )


def _parse_voice(segments: list[_Segment], state: _State) -> list[_Parsed]:
    """Parse every bar of one voice, in order, carrying the state along."""
    parsed: list[_Parsed] = []
    for segment in segments:
        parser = _BarParser(segment.text, state)
        parsed.append(_Parsed(parser.parse(), parser.bar_meter))
    return parsed


def _check_voice(
    voice: str, segments: list[_Segment], state: _State, multi_voice: bool
) -> tuple[int, Fraction, Fraction | None]:
    """Verify one voice's bars; return bars filled, total, and first-bar span."""
    label = f"voice {voice}, " if multi_voice else ""
    parsed = _parse_voice(segments, state)
    for index in range(len(parsed)):
        _check_bar(label, index, segments, parsed)
    filled = sum(max(bar.result.bars, 1) for bar in parsed)
    total = sum((bar.result.duration for bar in parsed), Fraction(0))
    first: Fraction | None = None
    if parsed:
        head = parsed[0]
        # A multi-measure rest opens on the downbeat like any full bar does.
        first = head.meter.bar if head.result.bars else head.result.duration
    return filled, total, first


def _check_voice_lengths(filled: dict[str, int]) -> None:
    """Require every voice to fill the same number of bars."""
    if len(set(filled.values())) <= 1:
        return
    described = ", ".join(f"voice {v}: {n} bars" for v, n in filled.items())
    raise ABCValidationError(
        f"the voices are different lengths ({described}); every voice "
        "must fill the same number of bars so they play together — a "
        "resting voice can use z rests or a Z multi-measure rest",
        ABCErrorCode.VOICE_MISALIGNED,
    )


def _check_voice_totals(totals: dict[str, Fraction], meter: _Meter) -> None:
    """Require every voice to hold the same total duration.

    Bar counts alone let a pickup in one voice slide the whole part out of
    phase against the others; the totals catch it.
    """
    full = meter.bar
    if full is None or len(set(totals.values())) <= 1:
        return
    described = ", ".join(f"voice {v}: {d / full} bars" for v, d in totals.items())
    raise ABCValidationError(
        f"the voices hold different amounts of music ({described}); they "
        "would drift apart as they play — give every voice the same total "
        "length, so a pickup in one is matched by a pickup or a rest in "
        "the others",
        ABCErrorCode.VOICE_MISALIGNED,
    )


def _check_voice_onsets(firsts: dict[str, Fraction | None]) -> None:
    """Require every voice's first bar to span the same duration.

    Equal totals still admit two pickups of different lengths — each
    complementing its own final bar, yet an eighth apart from the downbeat
    for the whole piece. The first bar is where the phase is set.
    """
    known = {v: d for v, d in firsts.items() if d is not None}
    if len(set(known.values())) <= 1:
        return
    described = ", ".join(f"voice {v}: {d}" for v, d in known.items())
    raise ABCValidationError(
        f"the voices open with different pickups ({described} of a whole "
        "note); every voice must enter the first barline together — give "
        "each voice the same pickup length, filling with rests if it "
        "waits",
        ABCErrorCode.VOICE_MISALIGNED,
    )


def _check_bar_durations(abc: str, headers: dict[str, str]) -> None:
    """Verify every voice's bar durations, and that voices stay aligned."""
    meter = _parse_meter(headers.get("M", ""))
    unit = _unit_length(headers.get("L", ""))
    voices = {v: _split_bars(text) for v, text in _voice_text(abc).items()}
    multi_voice = len(voices) > 1
    filled: dict[str, int] = {}
    totals: dict[str, Fraction] = {}
    firsts: dict[str, Fraction | None] = {}
    for voice, segments in voices.items():
        filled[voice], totals[voice], firsts[voice] = _check_voice(
            voice, segments, _State(meter, unit), multi_voice
        )
    if multi_voice:
        _check_voice_lengths(filled)
        _check_voice_totals(totals, meter)
        _check_voice_onsets(firsts)


def _check_required_headers(abc: str, headers: dict[str, str]) -> None:
    """Check the required fields are present, once, and in the right place."""
    missing = [h for h in REQUIRED_HEADERS if h not in headers]
    if missing:
        fields = ", ".join(f"{h}:" for h in missing)
        raise ABCValidationError(
            f"missing required header field(s): {fields}",
            ABCErrorCode.MISSING_HEADER,
        )
    first_line = abc.splitlines()[0].strip()
    if not first_line.startswith("X:"):
        raise ABCValidationError(
            "the first line must be the X: reference number",
            ABCErrorCode.BAD_FIRST_LINE,
        )
    tunes = sum(1 for line in abc.splitlines() if line.strip().startswith("X:"))
    if tunes > 1:
        raise ABCValidationError(
            f"there are {tunes} X: lines, so this is {tunes} tunes; "
            "return exactly one tune",
            ABCErrorCode.MULTIPLE_TUNES,
        )


def _check_key(value: str) -> None:
    """Check the ``K:`` value names a key a player could read."""
    if not _KEY.match(value.strip()):
        raise ABCValidationError(
            f"the K: header '{value}' is not a key i recognise; use a letter "
            "A-G with an optional # or b and an optional mode, like K:Ddor, "
            "K:F#m, K:Bb or K:Amix",
            ABCErrorCode.BAD_KEY,
        )


def _check_tempo(value: str) -> None:
    """Check the ``Q:`` value against the ABC tempo grammar."""
    text = _TEMPO_TEXT.sub(" ", value).strip()
    if not text or text.isdigit() or _TEMPO_BEATS.match(text):
        return
    raise ABCValidationError(
        f"the Q: header '{value}' is not a tempo i can read; write it as "
        'Q:1/4=96, Q:"andante" 3/8=60, or a bare beats-per-minute number',
        ABCErrorCode.BAD_TEMPO,
    )


def _check_midi_programs(abc: str) -> None:
    """Check every ``%%MIDI program`` sits in the general midi range."""
    for match in _MIDI_PROGRAM.finditer(abc):
        numbers = re.findall(r"-?\d+", match.group(1))
        if not numbers:
            raise ABCValidationError(
                "'%%MIDI program' needs a general midi program number from 0 to 127",
                ABCErrorCode.BAD_MIDI_PROGRAM,
            )
        program = int(numbers[-1])
        if not 0 <= program <= 127:
            raise ABCValidationError(
                f"'%%MIDI program {program}' is out of range; general midi "
                "programs run from 0 to 127",
                ABCErrorCode.BAD_MIDI_PROGRAM,
            )


def _check_body(abc: str) -> None:
    """Check there is music under the headers, and that it has barlines."""
    body = music_lines(abc)
    if not body:
        raise ABCValidationError(
            "the score has headers but no music lines", ABCErrorCode.NO_BODY
        )
    if not any("|" in line for line in body):
        raise ABCValidationError(
            "the music lines contain no bar lines ('|')", ABCErrorCode.NO_BARLINES
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
        With a human-readable reason when the tune is malformed, and an
        :class:`ABCErrorCode` on ``.code``. The message is written to be
        shown back to the model so it can self-correct.
    """
    stripped = abc.strip()
    if not stripped:
        raise ABCValidationError("the score is empty", ABCErrorCode.EMPTY)
    if "```" in stripped:
        raise ABCValidationError(
            "the score contains markdown code fences; return raw ABC only",
            ABCErrorCode.FENCED,
        )

    headers = parse_headers(stripped)
    _check_required_headers(stripped, headers)
    _check_key(headers["K"])
    if "Q" in headers:
        _check_tempo(headers["Q"])
    _check_midi_programs(stripped)
    _check_body(stripped)
    _check_bar_durations(stripped, headers)
