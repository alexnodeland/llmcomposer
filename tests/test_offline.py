"""The offline baseline is the study's control, so it is pinned exactly.

The golden string below is the whole point: if it still matches, the null
hypothesis is reproducible from a prompt in any interpreter, on any day.
"""

import re

from llmcomposer.abc_notation import parse_headers, validate_abc, voice_bars
from llmcomposer.offline import _BASE_SCALES, _compose, _transpose

GOLDEN_PROMPT = "a trio like rain on a window, wistful"

GOLDEN_ABC = (
    "X:1\n"
    "T:rainfall\n"
    "M:4/4\n"
    "L:1/8\n"
    "Q:1/4=66\n"
    "%%score [1 2 3]\n"
    'V:1 name="flute" sname="fl." clef=treble\n'
    "%%MIDI program 73\n"
    'V:2 name="harp" sname="hp." clef=bass\n'
    "%%MIDI program 46\n"
    'V:3 name="cello" sname="vc." clef=bass\n'
    "%%MIDI program 42\n"
    "K:Am\n"
    "V:1\n"
    "d2 Bc d2 e2 | d2 c2 BA Bc | dc BA cB cB | AB dc Bc Bc |\n"
    "AA A2 B2 cd | fg ed B2 cd | Bc A2 AA AA | e4 A4 |]\n"
    "V:2\n"
    "A,2 C2 E2 C2 | A,2 C2 E2 C2 | A,2 C2 E2 C2 | A,2 C2 E2 C2 |\n"
    "A,2 C2 E2 C2 | A,2 C2 E2 C2 | A,2 C2 E2 C2 | A,8 |]\n"
    "V:3\n"
    "A,,4 E,4 | A,,4 E,4 | A,,4 E,4 | A,,4 E,4 |\n"
    "A,,4 E,4 | A,,4 E,4 | A,,4 E,4 | A,,8 |]\n"
)

PROMPTS = [
    GOLDEN_PROMPT,
    "brutal fast metal in 7/8",
    "a quiet waltz in 3/4",
    "something for a garden at dawn",
    "ocean drift, still and slow",
    "bright morning, the whole band",
    "make it slower",
    "",
]

_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE = re.compile(r"([_^=]*)([A-Ga-g])([,']*)")


def _midi(token: str) -> int:
    """Convert a bare ABC note token to a MIDI pitch (``C`` is 60)."""
    match = _NOTE.fullmatch(token)
    assert match is not None, token
    accidental, letter, marks = match.groups()
    octave = (4 if letter.isupper() else 5) + marks.count("'") - marks.count(",")
    shift = accidental.count("^") - accidental.count("_")
    return (octave + 1) * 12 + _SEMITONES[letter.upper()] + shift


def _pitches(bars: list[str]) -> list[int]:
    """Every sounding pitch in a voice's bars, as MIDI numbers."""
    return [_midi(m.group(0)) for bar in bars for m in _NOTE.finditer(bar)]


def test_golden_score_is_byte_identical():
    reply, abc = _compose(GOLDEN_PROMPT, None)
    assert abc == GOLDEN_ABC
    assert reply == (
        "baseline: Am at 66 bpm, chosen by keyword match from a small mood "
        "table — no musical understanding, just the floor to beat."
    )


def test_every_prompt_yields_a_valid_tune():
    for prompt in PROMPTS:
        _, abc = _compose(prompt, None)
        validate_abc(abc)


def test_reply_describes_the_control_as_the_keyword_match_it_is():
    for prompt in PROMPTS:
        reply, _ = _compose(prompt, None)
        assert reply.startswith("baseline:")
        assert "i heard" not in reply
        # what it does: reads a few words, understands none of them.
        assert "keyword match" in reply
        assert "no musical understanding" in reply
        # and not the overclaim it used to make in the other direction.
        assert "ignores your words" not in reply
        assert "does not read your words" not in reply
        # the prompt is never echoed back at the collaborator
        if prompt.strip():
            assert prompt.strip() not in reply
        # lowercase house voice, apart from the key name it has to spell
        prose = reply
        for key in _BASE_SCALES:
            prose = prose.replace(key, "")
        assert prose == prose.lower()
        assert reply.count(".") == 1  # one sentence, no essay


def test_flute_voice_never_drops_below_middle_c():
    for prompt in PROMPTS:
        _, abc = _compose(prompt, None)
        melody = _pitches(voice_bars(abc)["1"])
        assert melody, abc
        assert min(melody) >= 60, f"{prompt!r} put the flute at {min(melody)}"


def test_voices_are_scored_with_clefs_short_names_and_a_bracket():
    _, abc = _compose("a trio, together", None)
    assert '\nV:3 name="cello" sname="vc." clef=bass\n' in abc
    assert '\nV:1 name="flute" sname="fl." clef=treble\n' in abc
    assert "\n%%score [1 2 3]\n" in abc
    assert parse_headers(abc)["K"] == "Ddor"


def test_the_bracket_is_declared_before_the_voices_it_brackets():
    # abcjs draws %%score around the staves declared after it; emitted below
    # the V: lines it brackets the last staff alone, which is what round 2
    # measured in the screenshots.
    for prompt in PROMPTS:
        _, abc = _compose(prompt, None)
        assert abc.index("%%score") < abc.index("V:1")


def test_the_harp_is_written_on_a_staff_it_actually_sits_on():
    _, abc = _compose("a trio, together", None)
    assert '\nV:2 name="harp" sname="hp." clef=bass\n' in abc
    harp = _pitches(voice_bars(abc)["2"])
    # a bass staff spans G2-A3 (43-57); nothing may need ledger lines.
    assert 43 <= min(harp) and max(harp) <= 57, harp


def test_lower_voices_stay_below_the_melody():
    _, abc = _compose("a trio, together", None)
    voices = voice_bars(abc)
    flute, harp, cello = (_pitches(voices[v]) for v in ("1", "2", "3"))
    assert max(cello) < min(harp)
    assert max(harp) < min(flute)
    assert min(cello) >= 36  # the cello's low C


def test_transpose_round_trips_through_octave_spellings():
    assert _transpose("C", -1) == "C,"
    assert _transpose("C,", -1) == "C,,"
    assert _transpose("c", -1) == "C"
    assert _transpose("^f", 1) == "^f'"
    assert _transpose("A,,", 2) == "A"
