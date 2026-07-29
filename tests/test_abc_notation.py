import pytest

from llmcomposer.abc_notation import (
    ABCValidationError,
    music_lines,
    parse_headers,
    validate_abc,
)

GOOD = (
    "X:1\nT:small hours\nM:4/4\nL:1/8\nQ:1/4=76\nK:Ddor\n"
    "DE FG A2 Bc | d2 cB AG FE | D2 E2 F2 G2 | A8 |]\n"
)


def test_good_abc_validates():
    validate_abc(GOOD)


def test_parse_headers():
    headers = parse_headers(GOOD)
    assert headers["X"] == "1"
    assert headers["K"] == "Ddor"
    assert headers["Q"] == "1/4=76"


def test_music_lines_excludes_headers():
    lines = music_lines(GOOD)
    assert len(lines) == 1
    assert lines[0].startswith("DE")


@pytest.mark.parametrize(
    ("abc", "fragment"),
    [
        ("", "empty"),
        ("```abc\nX:1\nK:C\nCDEF|\n```", "fences"),
        ("X:1\nT:x\nK:C\nCDEF |\n", "missing required header"),
        ("T:x\nX:1\nM:4/4\nL:1/8\nK:C\nCDEF |\n", "first line"),
        ("X:1\nM:4/4\nL:1/8\nK:C\n", "no music lines"),
        ("X:1\nM:4/4\nL:1/8\nK:C\nCDEF GABc\n", "bar lines"),
    ],
)
def test_bad_abc_rejected(abc: str, fragment: str):
    with pytest.raises(ABCValidationError, match=fragment):
        validate_abc(abc)


class TestParser:
    def test_duration_suffixes(self):
        from fractions import Fraction

        from llmcomposer.abc_notation import _duration

        assert _duration("") == 1
        assert _duration("2") == 2
        assert _duration("3/2") == Fraction(3, 2)
        assert _duration("/") == Fraction(1, 2)
        assert _duration("/2") == Fraction(1, 2)
        assert _duration("//") == Fraction(1, 4)

    def test_unparseable_token_rejected(self):
        abc = "X:1\nT:x\nM:4/4\nL:1/8\nK:C\nC2 D2 E2 F# |\n"
        with pytest.raises(ABCValidationError, match="unparseable ABC"):
            validate_abc(abc)

    def test_overfull_bar_rejected(self):
        abc = "X:1\nT:x\nM:4/4\nL:1/8\nK:C\nC2 D2 E2 F2 G2 | C8 |]\n"
        with pytest.raises(ABCValidationError, match="bar 1"):
            validate_abc(abc)

    def test_underfull_interior_bar_rejected(self):
        abc = "X:1\nT:x\nM:4/4\nL:1/8\nK:C\nC8 | D4 | E8 | C8 |]\n"
        with pytest.raises(ABCValidationError, match="bar 2"):
            validate_abc(abc)

    def test_pickup_and_final_bars_may_be_short(self):
        abc = "X:1\nT:x\nM:4/4\nL:1/8\nK:C\nGA | C2 E2 G2 c2 | C6 |]\n"
        validate_abc(abc)

    def test_triplets_count_correctly(self):
        abc = "X:1\nT:x\nM:4/4\nL:1/8\nK:C\n(3CDE F2 G2 A2 | C8 |]\n"
        validate_abc(abc)

    def test_chords_and_chord_symbols(self):
        abc = 'X:1\nT:x\nM:4/4\nL:1/8\nK:C\n"C"[CEG]2 [DF]2 "G"G2 B2 | C8 |]\n'
        validate_abc(abc)

    def test_rests_and_multimeasure_rests(self):
        abc = "X:1\nT:x\nM:4/4\nL:1/8\nK:C\nz2 C2 z4 | Z | C8 |]\n"
        validate_abc(abc)

    def test_repeats_allow_partial_bars(self):
        abc = "X:1\nT:x\nM:4/4\nL:1/8\nK:C\n|: C4 :| D8 | C8 |]\n"
        validate_abc(abc)

    def test_bad_meter_header(self):
        abc = "X:1\nT:x\nM:waltz\nL:1/8\nK:C\nC8 |]\n"
        with pytest.raises(ABCValidationError, match="not a valid meter"):
            validate_abc(abc)

    def test_six_eight_meter(self):
        abc = "X:1\nT:x\nM:6/8\nL:1/8\nK:G\nGAB dBG | G3 g3 |]\n"
        validate_abc(abc)


MULTI = (
    "X:1\nT:duet\nM:4/4\nL:1/8\nQ:1/4=76\n"
    'V:1 name="flute"\n%%MIDI program 73\n'
    'V:2 name="cello"\n%%MIDI program 42\n'
    "K:Am\n"
    "V:1\nAB cd e2 a2 | e2 c2 A4 |]\n"
    "V:2\nA,4 E,4 | A,8 |]\n"
)


class TestVoices:
    def test_multi_voice_tune_validates(self):
        validate_abc(MULTI)

    def test_voice_bars_grouping(self):
        from llmcomposer.abc_notation import voice_bars

        voices = voice_bars(MULTI)
        assert set(voices) == {"1", "2"}
        assert len(voices["1"]) == 2
        assert len(voices["2"]) == 2

    def test_mismatched_voice_lengths_rejected(self):
        bad = MULTI.replace("V:2\nA,4 E,4 | A,8 |]", "V:2\nA,8 |]")
        with pytest.raises(ABCValidationError, match="different lengths"):
            validate_abc(bad)

    def test_bad_bar_names_the_voice(self):
        bad = MULTI.replace("A,4 E,4 |", "A,4 E,4 C4 |")
        with pytest.raises(ABCValidationError, match="voice 2, bar 1"):
            validate_abc(bad)


def test_dynamics_and_articulations_validate():
    abc = (
        "X:1\nT:x\nM:4/4\nL:1/8\nK:C\n!p!.C2 D2 ~E2 uF2 | !f!(GA) (3cde f2 c2 | C8 |]\n"
    )
    validate_abc(abc)
