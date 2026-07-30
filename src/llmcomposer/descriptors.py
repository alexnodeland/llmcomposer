"""Symbolic descriptors: what the notes actually do.

The validator in :mod:`llmcomposer.abc_notation` answers *is this
well-formed ABC?*. These descriptors answer *what music is it?* — pitch
content against the declared key, melodic shape, repetition, register
against each voice's General MIDI instrument, and the closing cadence.
They are the measurement side of the project: cheap, standard, and
comparable across models and against the baseline.

Pitch decoding lives here rather than in :mod:`llmcomposer.abc_notation`
because the validator only ever needed durations. Everything is
best-effort: a token this module cannot read is skipped rather than
raised on, so a descriptor row exists for every score an eval run
collects — including the malformed ones.

Conventions
-----------
Durations are :class:`~fractions.Fraction` throughout (in unit-note
lengths) and are only converted to ``float`` at the boundary, when a
statistic is reported. Pitches are MIDI numbers with middle C (ABC
``C``) at 60. A single ``K:`` is assumed for the whole tune; inline key
changes are ignored.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from typing import NamedTuple

from pydantic import BaseModel, Field

from .abc_notation import parse_headers, voice_bars

__all__ = [
    "GM_RANGES",
    "GMRange",
    "ScoreDescriptors",
    "describe",
    "key_signature",
]

# --------------------------------------------------------------------------
# key signatures and modes
# --------------------------------------------------------------------------

_LETTER_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NATURAL_FIFTHS = {"F": -1, "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5}
_SHARP_ORDER = "FCGDAEB"
_FLAT_ORDER = "BEADGCF"

# How far, in fifths, a mode's tonic sits above its parent major key.
# D dorian shares C major's signature: fifths(D major) - 2 = 0.
_MODE_FIFTHS = {
    "maj": 0,
    "ion": 0,
    "mix": 1,
    "dor": 2,
    "min": 3,
    "aeo": 3,
    "m": 3,
    "phr": 4,
    "loc": 5,
    "lyd": -1,
}
_MODE_NAMES = {
    "maj": "major",
    "ion": "ionian",
    "mix": "mixolydian",
    "dor": "dorian",
    "min": "minor",
    "aeo": "aeolian",
    "m": "minor",
    "phr": "phrygian",
    "loc": "locrian",
    "lyd": "lydian",
}
_MINOR_MODES = frozenset({"min", "aeo", "m", "dor", "phr", "loc"})

# The degree that tells a mode apart from the major or minor scale it would
# otherwise collapse into, in semitones above the tonic: the characteristic
# degree first, then the degree whose presence would mean the collapse.
_CHARACTERISTIC = {
    "maj": (11, 10),  # natural 7 rather than a mixolydian flat 7
    "ion": (11, 10),
    "mix": (10, 11),  # flat 7 rather than a major leading tone
    "dor": (9, 8),  # natural 6 rather than a minor flat 6
    "min": (8, 9),  # flat 6 rather than a dorian natural 6
    "aeo": (8, 9),
    "m": (8, 9),
    "phr": (1, 2),  # flat 2 rather than a natural second
    "loc": (6, 7),  # flat 5 rather than a perfect fifth
    "lyd": (6, 5),  # sharp 4 rather than a natural fourth
}

_PC_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")

# Krumhansl & Kessler (1982) key profiles, in probe-tone rating units.
_PROFILE_MAJOR = (
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
)  # fmt: skip
_PROFILE_MINOR = (
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
)  # fmt: skip

_KEY_HEAD = re.compile(r"^\s*([A-Ga-g])\s*([#b♯♭]?)\s*([A-Za-z]*)")


class _Key(NamedTuple):
    """A parsed ``K:`` header: tonic pitch class, mode key, fifths."""

    root_pc: int | None
    mode: str
    fifths: int


def _parse_key(text: str) -> _Key:
    """Parse a ``K:`` value into tonic, mode, and position on the circle."""
    match = _KEY_HEAD.match(text.strip())
    if match is None:
        return _Key(None, "maj", 0)
    letter = match.group(1).upper()
    accidental = match.group(2)
    word = match.group(3).lower()
    mode = word[:3] if word[:3] in _MODE_FIFTHS else ("m" if word == "m" else "maj")
    shift = 1 if accidental in {"#", "♯"} else -1 if accidental in {"b", "♭"} else 0
    root_pc = (_LETTER_SEMITONE[letter] + shift) % 12
    fifths = _NATURAL_FIFTHS[letter] + 7 * shift - _MODE_FIFTHS[mode]
    return _Key(root_pc, mode, fifths)


def key_signature(key: str) -> dict[str, int]:
    """Return the accidental each letter carries in a key.

    Parameters
    ----------
    key : str
        A ``K:`` header value such as ``"C"``, ``"Ddor"``, or ``"Bbmaj"``.

    Returns
    -------
    dict[str, int]
        Uppercase note letters mapped to their semitone alteration
        (``+1`` sharp, ``-1`` flat); letters absent are natural.
    """
    fifths = _parse_key(key).fifths
    if fifths > 0:
        return {letter: 1 for letter in _SHARP_ORDER[: min(fifths, 7)]}
    return {letter: -1 for letter in _FLAT_ORDER[: min(-fifths, 7)]}


# --------------------------------------------------------------------------
# General MIDI playable ranges
# --------------------------------------------------------------------------


class GMRange(NamedTuple):
    """A General MIDI program's name and its playable MIDI pitch range."""

    name: str
    low: int
    high: int


GM_RANGES: dict[int, GMRange] = {
    0: GMRange("acoustic grand piano", 21, 108),
    1: GMRange("bright acoustic piano", 21, 108),
    4: GMRange("electric piano", 28, 103),
    6: GMRange("harpsichord", 29, 89),
    8: GMRange("celesta", 60, 108),
    9: GMRange("glockenspiel", 79, 108),
    11: GMRange("vibraphone", 53, 89),
    13: GMRange("xylophone", 65, 108),
    14: GMRange("tubular bells", 60, 89),
    19: GMRange("church organ", 36, 96),
    21: GMRange("accordion", 41, 89),
    22: GMRange("harmonica", 60, 96),
    24: GMRange("nylon guitar", 40, 83),
    25: GMRange("steel guitar", 40, 83),
    26: GMRange("jazz guitar", 40, 86),
    27: GMRange("clean electric guitar", 40, 86),
    32: GMRange("acoustic bass", 28, 60),
    33: GMRange("fingered electric bass", 28, 67),
    40: GMRange("violin", 55, 103),
    41: GMRange("viola", 48, 91),
    42: GMRange("cello", 36, 81),
    43: GMRange("contrabass", 28, 60),
    45: GMRange("pizzicato strings", 36, 96),
    46: GMRange("orchestral harp", 24, 103),
    47: GMRange("timpani", 40, 55),
    48: GMRange("string ensemble", 28, 96),
    52: GMRange("choir aahs", 43, 84),
    56: GMRange("trumpet", 54, 82),
    57: GMRange("trombone", 34, 72),
    58: GMRange("tuba", 28, 58),
    60: GMRange("french horn", 34, 77),
    64: GMRange("soprano sax", 56, 87),
    65: GMRange("alto sax", 49, 81),
    66: GMRange("tenor sax", 44, 76),
    67: GMRange("baritone sax", 36, 69),
    68: GMRange("oboe", 58, 91),
    69: GMRange("english horn", 52, 84),
    70: GMRange("bassoon", 34, 75),
    71: GMRange("clarinet", 50, 91),
    72: GMRange("piccolo", 74, 108),
    73: GMRange("flute", 60, 96),
    74: GMRange("recorder", 60, 96),
    75: GMRange("pan flute", 60, 96),
    79: GMRange("ocarina", 72, 91),
    105: GMRange("banjo", 48, 79),
    110: GMRange("fiddle", 55, 103),
}
"""Playable MIDI ranges for the General MIDI programs the agent is told to use."""


# --------------------------------------------------------------------------
# tokenizing a voice into timed events
# --------------------------------------------------------------------------

_NEUTRAL = re.compile(
    r'"[^"]*"|![^!]*!|\{[^}]*\}|\[[A-Za-z]:[^\]]*\]|y\d*|[<>\-\\)]'
    r"|\((?!\d)|[.~HLMOPSTuv]"
)
_VOLTA = re.compile(r"^\s*\[?\d[\s,-]*")
_DURATION = re.compile(r"^(\d+)?((?:/\d*)*)$")
_NOTE_TOKEN = re.compile(
    r"(?P<acc>[_^=]{0,2})(?P<letter>[A-Ga-g])(?P<oct>[,']*)"
    r"(?P<dur>\d*(?:/\d*)*)"
)
_REST = re.compile(r"[zx](\d*(?:/\d*)*)")
_BIG_REST = re.compile(r"[ZX](\d*)")
_CHORD = re.compile(r"\[([^\]]*)\](\d*(?:/\d*)*)")
_TUPLET = re.compile(r"\((\d)(?::(\d?))?(?::(\d?))?")
_TUPLET_TIME = {2: 3, 3: 2, 4: 3, 6: 2, 8: 3}
_ALTER = {"^": 1, "^^": 2, "_": -1, "__": -2, "=": 0, "": 0}

_VOICE_DECL = re.compile(r"^V:\s*(\S+)")
_VOICE_NAME = re.compile(r'(?:name|nm)="([^"]*)"')
_MIDI_PROGRAM = re.compile(r"^%%MIDI\s+program(?:\s+\d+)?\s+(\d+)")


def _duration(text: str) -> Fraction:
    """Read an ABC duration suffix; unreadable suffixes count as one unit."""
    match = _DURATION.match(text)
    if match is None:
        return Fraction(1)
    value = Fraction(int(match.group(1))) if match.group(1) else Fraction(1)
    for denominator in re.findall(r"/(\d*)", match.group(2) or ""):
        value /= int(denominator) if denominator else 2
    return value


@dataclass
class _Event:
    """One sounding moment in a voice: when, how long, which pitches."""

    bar: int
    onset: Fraction
    duration: Fraction
    pitches: tuple[int, ...] = ()

    @property
    def end(self) -> Fraction:
        """The event's release point, in unit-note lengths."""
        return self.onset + self.duration


@dataclass
class _VoiceReader:
    """Walks one voice's bars, emitting timed :class:`_Event` objects."""

    signature: dict[str, int]
    full_bar: Fraction
    events: list[_Event] = field(default_factory=list["_Event"])
    clock: Fraction = Fraction(0)
    bar: int = 0
    accidentals: dict[tuple[str, int], int] = field(
        default_factory=dict[tuple[str, int], int]
    )
    tuplet_factor: Fraction = Fraction(1)
    tuplet_left: int = 0

    def read(self, bars: Sequence[str]) -> list[_Event]:
        """Read every bar of the voice and return its events in order."""
        for index, bar in enumerate(bars):
            self.bar = index
            self.accidentals.clear()
            self._read_bar(bar)
        return self.events

    def _read_bar(self, bar: str) -> None:
        text = _NEUTRAL.sub(" ", _VOLTA.sub("", bar))
        pos = 0
        while pos < len(text):
            if text[pos].isspace():
                pos += 1
                continue
            advanced = self._token(text, pos)
            pos = advanced if advanced is not None else pos + 1

    def _token(self, text: str, pos: int) -> int | None:
        for regex, handler in (
            (_TUPLET, self._on_tuplet),
            (_CHORD, self._on_chord),
            (_NOTE_TOKEN, self._on_note),
            (_REST, self._on_rest),
            (_BIG_REST, self._on_big_rest),
        ):
            match = regex.match(text, pos)
            if match:
                handler(match)
                return match.end()
        return None

    def _emit(self, duration: Fraction, pitches: tuple[int, ...]) -> None:
        if self.tuplet_left > 0:
            duration *= self.tuplet_factor
            self.tuplet_left -= 1
        self.events.append(_Event(self.bar, self.clock, duration, pitches))
        self.clock += duration

    def _pitch(self, match: re.Match[str]) -> int:
        letter = match.group("letter")
        upper = letter.upper()
        octave = 5 if letter.islower() else 4
        octave += match.group("oct").count("'") - match.group("oct").count(",")
        written = _ALTER.get(match.group("acc"))
        if match.group("acc"):
            alter = written if written is not None else 0
            self.accidentals[(upper, octave)] = alter
        else:
            alter = self.accidentals.get((upper, octave), self.signature.get(upper, 0))
        return 12 * (octave + 1) + _LETTER_SEMITONE[upper] + alter

    def _on_note(self, match: re.Match[str]) -> None:
        self._emit(_duration(match.group("dur")), (self._pitch(match),))

    def _on_chord(self, match: re.Match[str]) -> None:
        inner = list(_NOTE_TOKEN.finditer(match.group(1)))
        if not inner:
            return
        pitches = tuple(self._pitch(note) for note in inner)
        longest = max(_duration(note.group("dur")) for note in inner)
        self._emit(longest * _duration(match.group(2)), pitches)

    def _on_rest(self, match: re.Match[str]) -> None:
        self._emit(_duration(match.group(1)), ())

    def _on_big_rest(self, match: re.Match[str]) -> None:
        self._emit(int(match.group(1) or 1) * self.full_bar, ())

    def _on_tuplet(self, match: re.Match[str]) -> None:
        notes = int(match.group(1))
        stated = match.group(2)
        time = int(stated) if stated else _TUPLET_TIME.get(notes)
        if time is None:
            return
        self.tuplet_factor = Fraction(time, notes)
        self.tuplet_left = int(match.group(3) or notes)


def _voice_metadata(abc: str) -> tuple[dict[str, str], dict[str, int]]:
    """Return each voice's declared name and General MIDI program."""
    names: dict[str, str] = {}
    programs: dict[str, int] = {}
    current = "1"
    for raw in abc.splitlines():
        line = raw.strip()
        declaration = _VOICE_DECL.match(line)
        if declaration:
            current = declaration.group(1)
            named = _VOICE_NAME.search(line)
            if named:
                names[current] = named.group(1)
            continue
        program = _MIDI_PROGRAM.match(line)
        if program:
            programs[current] = int(program.group(1))
    return names, programs


# --------------------------------------------------------------------------
# statistics over the event stream
# --------------------------------------------------------------------------


def _meter(headers: dict[str, str]) -> Fraction:
    """Bar length as a fraction of a whole note; 4/4 when unreadable."""
    meter = headers.get("M", "").strip()
    if meter in {"C", "c"}:
        return Fraction(4, 4)
    if meter in {"C|", "c|"}:
        return Fraction(2, 2)
    match = re.match(r"^(\d+)/(\d+)$", meter)
    return Fraction(int(match.group(1)), int(match.group(2))) if match else Fraction(1)


def _unit(headers: dict[str, str]) -> Fraction:
    """Return the ``L:`` unit note length; an eighth when unreadable."""
    match = re.match(r"^(\d+)/(\d+)$", headers.get("L", "").strip())
    return (
        Fraction(int(match.group(1)), int(match.group(2))) if match else Fraction(1, 8)
    )


def _pitch_class_weights(events: Sequence[_Event]) -> list[Fraction]:
    """Duration-weighted mass on each of the twelve pitch classes."""
    weights = [Fraction(0)] * 12
    for event in events:
        for pitch in event.pitches:
            weights[pitch % 12] += event.duration
    return weights


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation; ``0.0`` when either series is constant."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denominator = math.sqrt(sum(d * d for d in dx) * sum(d * d for d in dy))
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / denominator


def _profile(mode: str) -> tuple[float, ...]:
    """Return the Krumhansl-Kessler profile appropriate to a mode."""
    return _PROFILE_MINOR if mode in _MINOR_MODES else _PROFILE_MAJOR


def _rotate(profile: Sequence[float], root_pc: int) -> list[float]:
    """Rotate a pitch-class profile so its tonic sits at ``root_pc``."""
    return [profile[(pc - root_pc) % 12] for pc in range(12)]


def _key_correlation(weights: Sequence[Fraction], key: _Key) -> float:
    """Correlate realized pitch content with the declared key's profile.

    The Krumhansl-Kessler profiles are major and minor only, so this is a
    tonal-fit number: a modal tune is measured against the major or minor
    template nearest its mode, never against the mode itself. Mode
    adherence is :func:`_diatonic_adherence` and :func:`_mode_match`.
    """
    if key.root_pc is None or not any(weights):
        return 0.0
    observed = [float(w) for w in weights]
    return _pearson(observed, _rotate(_profile(key.mode), key.root_pc))


def _scale_pcs(mode: str, root_pc: int) -> frozenset[int]:
    """Return the seven pitch classes of ``mode`` built on ``root_pc``.

    The diatonic set is a run of seven fifths, rotated so the mode's own
    tonic sits where the parent major's would.
    """
    shift = _MODE_FIFTHS.get(mode, 0)
    return frozenset((root_pc + 7 * (fifth - shift)) % 12 for fifth in range(-1, 6))


def _diatonic_adherence(weights: Sequence[Fraction], key: _Key) -> float | None:
    """Share of the pitch-class mass sitting inside the declared mode."""
    total = sum(weights, Fraction(0))
    if key.root_pc is None or total == 0:
        return None
    scale = _scale_pcs(key.mode, key.root_pc)
    inside = sum(
        (weight for pc, weight in enumerate(weights) if pc in scale), Fraction(0)
    )
    return float(inside / total)


def _mode_match(weights: Sequence[Fraction], key: _Key) -> bool | None:
    """Report whether the mode's characteristic degree is the one sung.

    Dorian is dorian because of its natural sixth, mixolydian because of
    its flat seventh. The test weighs that degree against the one that
    would mean the tune had collapsed into plain major or minor. When
    neither degree sounds at all the pitch content decides nothing, so the
    answer is ``None`` (undecided) rather than a failure — a hexatonic
    modal tune that contradicts nothing is not a mode failure.
    """
    degrees = _CHARACTERISTIC.get(key.mode)
    if key.root_pc is None or degrees is None or not any(weights):
        return None
    characteristic, rival = degrees
    upheld = weights[(key.root_pc + characteristic) % 12]
    collapsed = weights[(key.root_pc + rival) % 12]
    if upheld == 0 and collapsed == 0:
        return None
    return upheld > collapsed


def _infer_key(weights: Sequence[Fraction]) -> tuple[str, float]:
    """Best-fitting major/minor key and its correlation, Krumhansl-style."""
    if not any(weights):
        return "", 0.0
    observed = [float(w) for w in weights]
    best_name = ""
    best_score = -2.0
    for root_pc in range(12):
        for suffix, profile in (("", _PROFILE_MAJOR), ("m", _PROFILE_MINOR)):
            score = _pearson(observed, _rotate(profile, root_pc))
            if score > best_score:
                best_score, best_name = score, f"{_PC_NAMES[root_pc]}{suffix}"
    return best_name, best_score


def _entropy(weights: Sequence[Fraction]) -> float:
    """Shannon entropy in bits of the pitch-class distribution."""
    total = sum(weights, Fraction(0))
    if total == 0:
        return 0.0
    bits = 0.0
    for weight in weights:
        if weight:
            p = float(weight / total)
            bits -= p * math.log2(p)
    return bits


def _melody_pitches(events: Sequence[_Event]) -> list[int]:
    """Return the top note of every sounding event, rests excluded."""
    return [max(event.pitches) for event in events if event.pitches]


def _repetition_rate(pitches: Sequence[int], n: int = 3) -> float:
    """Fraction of pitch ``n``-grams that are not first occurrences."""
    grams = [tuple(pitches[i : i + n]) for i in range(len(pitches) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def _interval_stats(pitches: Sequence[int]) -> tuple[float, float]:
    """Mean absolute melodic interval and the fraction that are leaps."""
    steps = [abs(b - a) for a, b in pairwise(pitches)]
    if not steps:
        return 0.0, 0.0
    return sum(steps) / len(steps), sum(s > 4 for s in steps) / len(steps)


def _event_at(events: Sequence[_Event], time: Fraction) -> _Event | None:
    """Return the event sounding at ``time`` (``None`` during a rest)."""
    for event in events:
        if event.onset <= time < event.end:
            return event
    return None


def _volta_bars(voices: dict[str, list[str]]) -> frozenset[int]:
    """Bar indices that open a repeat ending, in any voice."""
    return frozenset(
        index
        for bars in voices.values()
        for index, bar in enumerate(bars)
        if _VOLTA.match(bar)
    )


def _pair_timeline(
    upper: Sequence[_Event],
    lower: Sequence[_Event],
    volta_bars: frozenset[int] = frozenset(),
) -> list[list[tuple[int, int]]]:
    """Sample two voices at every shared onset, in runs cut at volta seams.

    A written score is not a performance: the last chord of a first ending
    and the first chord of a second ending never sound one after the
    other, because the repeat sits between them. So the walk starts a
    fresh run at any bar boundary that touches a volta, and motion is only
    ever read within a run.
    """
    onsets = sorted({event.onset for event in (*upper, *lower)})
    runs: list[list[tuple[int, int]]] = [[]]
    previous: frozenset[int] | None = None
    for time in onsets:
        top = _event_at(upper, time)
        bottom = _event_at(lower, time)
        if top is None or bottom is None or not top.pitches or not bottom.pitches:
            continue
        bars = frozenset({top.bar, bottom.bar})
        if previous is not None and bars != previous and (bars | previous) & volta_bars:
            runs.append([])
        runs[-1].append((max(top.pitches), min(bottom.pitches)))
        previous = bars
    return runs


def _parallels(runs: Sequence[Sequence[tuple[int, int]]]) -> tuple[int, int]:
    """Count parallel fifths and octaves in a two-voice simultaneity walk."""
    fifths = octaves = 0
    for run in runs:
        for (u0, l0), (u1, l1) in pairwise(run):
            moved_up = u1 - u0
            moved_low = l1 - l0
            if moved_up == 0 or moved_low == 0 or (moved_up > 0) != (moved_low > 0):
                continue
            before = abs(u0 - l0) % 12
            after = abs(u1 - l1) % 12
            if before == after == 7:
                fifths += 1
            elif before == after == 0:
                octaves += 1
    return fifths, octaves


def _crossings(
    voices: Sequence[Sequence[_Event]], volta_bars: frozenset[int] = frozenset()
) -> int:
    """Moments where an upper voice sinks below the voice beneath it."""
    total = 0
    for upper, lower in pairwise(voices):
        for run in _pair_timeline(upper, lower, volta_bars):
            total += sum(1 for top, bottom in run if top < bottom)
    return total


def _last_pitch_in_bar(events: Sequence[_Event], bar: int) -> int | None:
    """Return a bar's final sounding pitch, if the voice plays there."""
    sounded = [event for event in events if event.bar == bar and event.pitches]
    return min(sounded[-1].pitches) if sounded else None


_CADENCES = {
    (7, 0): "authentic",
    (5, 0): "plagal",
    (7, 9): "deceptive",
    (7, 8): "deceptive",
}


def _cadence(bass: Sequence[_Event], root_pc: int | None, bars: int) -> str:
    """Classify the close from the bass motion over the last two bars."""
    if root_pc is None or bars < 2:
        return "none"
    penult = _last_pitch_in_bar(bass, bars - 2)
    final = _last_pitch_in_bar(bass, bars - 1)
    if penult is None or final is None:
        return "none"
    degrees = ((penult - root_pc) % 12, (final - root_pc) % 12)
    if degrees in _CADENCES:
        return _CADENCES[degrees]
    return "half" if degrees[1] == 7 else "none"


def _range_violations(
    ranges: dict[str, tuple[int, int]],
    names: dict[str, str],
    programs: dict[str, int],
) -> list[str]:
    """Name every voice whose written pitches leave its instrument's range."""
    violations: list[str] = []
    for voice, (low, high) in ranges.items():
        program = programs.get(voice)
        playable = GM_RANGES.get(program) if program is not None else None
        if playable is None:
            continue
        label = f"voice {voice} ({names.get(voice, playable.name)}, program {program})"
        if low < playable.low:
            violations.append(
                f"{label}: lowest note {low} is below the instrument's "
                f"range {playable.low}-{playable.high}"
            )
        if high > playable.high:
            violations.append(
                f"{label}: highest note {high} is above the instrument's "
                f"range {playable.low}-{playable.high}"
            )
    return violations


# --------------------------------------------------------------------------
# the descriptor row
# --------------------------------------------------------------------------


class ScoreDescriptors(BaseModel):
    """Cheap, standard symbolic statistics for one ABC score.

    Every ratio is a ``float`` computed from exact
    :class:`~fractions.Fraction` durations and rounded to four places for
    stable serialization.
    """

    meter: str = Field(default="", description="The M: header as written.")
    declared_key: str = Field(default="", description="The K: header as written.")
    declared_mode: str = Field(
        default="major", description="Mode named by K: (major, dorian, …)."
    )
    inferred_key: str = Field(
        default="",
        description="Best-fitting major/minor key by Krumhansl-Kessler profile.",
    )
    key_correlation: float = Field(
        default=0.0,
        description="Krumhansl-Kessler fit of the realized pitch content to "
        "the DECLARED key, against the major or minor template nearest its "
        "mode. A tonal fit, not a test of the mode itself.",
    )
    inferred_key_correlation: float = Field(
        default=0.0, description="Profile correlation for the inferred key."
    )
    key_match: bool = Field(
        default=False, description="Whether inferred and declared tonics agree."
    )
    diatonic_adherence: float | None = Field(
        default=None,
        description="Duration-weighted share of the pitch-class mass falling "
        "inside the declared mode's own scale; None when K: names no tonic.",
    )
    mode_match: bool | None = Field(
        default=None,
        description="Whether the declared mode's characteristic degree "
        "(dorian's natural 6, mixolydian's flat 7, phrygian's flat 2, "
        "lydian's sharp 4, minor's flat 6) outweighs the degree that would "
        "collapse it to plain major or minor; None when K: names no tonic "
        "or when neither degree sounds (the content decides nothing).",
    )
    pitch_class_entropy: float = Field(
        default=0.0, description="Entropy (bits) of the pitch-class distribution."
    )
    ngram_repetition_rate: float = Field(
        default=0.0, description="Fraction of melodic pitch 3-grams that repeat."
    )
    mean_abs_interval: float = Field(
        default=0.0, description="Mean absolute melodic interval in semitones."
    )
    leap_rate: float = Field(
        default=0.0, description="Fraction of melodic intervals wider than 4 semitones."
    )
    rest_fraction: float = Field(
        default=0.0, description="Share of total written duration spent resting."
    )
    note_count: int = Field(default=0, description="Sounding events across all voices.")
    bar_count: int = Field(default=0, description="Bars in the longest voice.")
    voice_count: int = Field(default=0, description="Voices carrying music.")
    voice_ranges: dict[str, tuple[int, int]] = Field(
        default_factory=dict[str, tuple[int, int]],
        description="Voice id mapped to its (lowest, highest) MIDI pitch.",
    )
    voice_programs: dict[str, int] = Field(
        default_factory=dict[str, int],
        description="Voice id mapped to its declared General MIDI program.",
    )
    range_violations: list[str] = Field(
        default_factory=list[str],
        description="Voices written outside their instrument's playable range.",
    )
    voice_crossing_count: int = Field(
        default=0, description="Moments an upper voice sinks below its neighbour."
    )
    parallel_fifth_count: int = Field(
        default=0, description="Parallel fifths between the outer voices."
    )
    parallel_octave_count: int = Field(
        default=0, description="Parallel octaves between the outer voices."
    )
    cadence: str = Field(
        default="none",
        description="authentic, plagal, half, deceptive, or none (best-effort).",
    )


def _rest_fraction(events: Sequence[_Event]) -> float:
    """Rest duration over total duration, computed exactly then rounded."""
    total = sum((event.duration for event in events), Fraction(0))
    if total == 0:
        return 0.0
    rested = sum((event.duration for event in events if not event.pitches), Fraction(0))
    return float(rested / total)


def _voice_ranges(
    read: dict[str, list[_Event]],
) -> dict[str, tuple[int, int]]:
    """Lowest and highest MIDI pitch written for each voice."""
    ranges: dict[str, tuple[int, int]] = {}
    for voice, events in read.items():
        pitches = [pitch for event in events for pitch in event.pitches]
        if pitches:
            ranges[voice] = (min(pitches), max(pitches))
    return ranges


def describe(abc: str) -> ScoreDescriptors:
    """Compute symbolic descriptors for one ABC tune.

    Parameters
    ----------
    abc : str
        The ABC source of a single tune. Need not be valid: unreadable
        tokens are skipped so that malformed scores still yield a row.

    Returns
    -------
    ScoreDescriptors
        Pitch, shape, register, and cadence statistics for the tune.

    Notes
    -----
    Melodic statistics (repetition, intervals, leaps) are taken from the
    first declared voice — the melody by the agent's own contract — while
    pitch-class statistics pool every voice. Parallels are counted
    between the outer voices only, and never across a volta seam, where
    two bars are written next to each other but never heard that way;
    crossings between each adjacent pair. Key fit comes in two kinds:
    ``key_correlation`` is the major/minor template fit, while
    ``diatonic_adherence`` and ``mode_match`` ask whether the tune means
    the mode its ``K:`` names.
    """
    headers = parse_headers(abc)
    key = _parse_key(headers.get("K", ""))
    signature = key_signature(headers.get("K", ""))
    full_bar = _meter(headers) / _unit(headers)
    voices = voice_bars(abc)
    read = {
        voice: _VoiceReader(signature, full_bar).read(bars)
        for voice, bars in voices.items()
    }
    read = {voice: events for voice, events in read.items() if events}
    order = list(read)
    every = [event for events in read.values() for event in events]
    melody = _melody_pitches(read[order[0]]) if order else []
    weights = _pitch_class_weights(every)
    inferred, inferred_correlation = _infer_key(weights)
    mean_interval, leap_rate = _interval_stats(melody)
    ranges = _voice_ranges(read)
    names, programs = _voice_metadata(abc)
    volta = _volta_bars(voices)
    outer = (
        _pair_timeline(read[order[0]], read[order[-1]], volta) if len(order) > 1 else []
    )
    fifths, octaves = _parallels(outer)
    adherence = _diatonic_adherence(weights, key)
    bars = max((len(b) for b in voices.values()), default=0)
    return ScoreDescriptors(
        meter=headers.get("M", ""),
        declared_key=headers.get("K", ""),
        declared_mode=_MODE_NAMES.get(key.mode, "major"),
        inferred_key=inferred,
        key_correlation=round(_key_correlation(weights, key), 4),
        inferred_key_correlation=round(inferred_correlation, 4),
        key_match=bool(inferred)
        and _PC_NAMES.index(inferred.rstrip("m")) == key.root_pc,
        diatonic_adherence=None if adherence is None else round(adherence, 4),
        mode_match=_mode_match(weights, key),
        pitch_class_entropy=round(_entropy(weights), 4),
        ngram_repetition_rate=round(_repetition_rate(melody), 4),
        mean_abs_interval=round(mean_interval, 4),
        leap_rate=round(leap_rate, 4),
        rest_fraction=round(_rest_fraction(every), 4),
        note_count=sum(1 for event in every if event.pitches),
        bar_count=bars,
        voice_count=len(order),
        voice_ranges=ranges,
        voice_programs={v: p for v, p in programs.items() if v in read},
        range_violations=_range_violations(ranges, names, programs),
        voice_crossing_count=_crossings([read[voice] for voice in order], volta),
        parallel_fifth_count=fifths,
        parallel_octave_count=octaves,
        cadence=_cadence(read[order[-1]], key.root_pc, bars) if order else "none",
    )
