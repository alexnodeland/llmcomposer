from fractions import Fraction
from pathlib import Path

import pytest

from llmcomposer.abc_notation import (
    ABCErrorCode,
    ABCValidationError,
    music_lines,
    parse_headers,
    validate_abc,
    voice_bars,
)

GOOD = (
    "X:1\nT:small hours\nM:4/4\nL:1/8\nQ:1/4=76\nK:Ddor\n"
    "DE FG A2 Bc | d2 cB AG FE | D2 E2 F2 G2 | A8 |]\n"
)

HEAD = "X:1\nT:x\nM:4/4\nL:1/8\nK:C\n"

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.abc"))


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
    ("abc", "fragment", "code"),
    [
        ("", "empty", ABCErrorCode.EMPTY),
        ("```abc\nX:1\nK:C\nCDEF|\n```", "fences", ABCErrorCode.FENCED),
        (
            "X:1\nT:x\nK:C\nCDEF |\n",
            "missing required header",
            ABCErrorCode.MISSING_HEADER,
        ),
        (
            "T:x\nX:1\nM:4/4\nL:1/8\nK:C\nCDEF |\n",
            "first line",
            ABCErrorCode.BAD_FIRST_LINE,
        ),
        ("X:1\nM:4/4\nL:1/8\nK:C\n", "no music lines", ABCErrorCode.NO_BODY),
        ("X:1\nM:4/4\nL:1/8\nK:C\nCDEF GABc\n", "bar lines", ABCErrorCode.NO_BARLINES),
    ],
)
def test_bad_abc_rejected(abc: str, fragment: str, code: ABCErrorCode):
    with pytest.raises(ABCValidationError, match=fragment) as caught:
        validate_abc(abc)
    assert caught.value.code is code


def test_error_code_defaults_to_unparseable():
    assert ABCValidationError("nope").code is ABCErrorCode.UNPARSEABLE


def test_error_codes_are_snake_case_of_their_names():
    for member in ABCErrorCode:
        assert member.value == member.name.lower()


class TestParser:
    def test_duration_suffixes(self):
        from llmcomposer.abc_notation import _duration

        assert _duration("") == 1
        assert _duration("2") == 2
        assert _duration("3/2") == Fraction(3, 2)
        assert _duration("/") == Fraction(1, 2)
        assert _duration("/2") == Fraction(1, 2)
        assert _duration("//") == Fraction(1, 4)

    def test_unparseable_token_rejected(self):
        abc = HEAD + "C2 D2 E2 F# |\n"
        with pytest.raises(ABCValidationError, match="unparseable ABC") as caught:
            validate_abc(abc)
        assert caught.value.code is ABCErrorCode.UNPARSEABLE

    def test_overfull_bar_rejected(self):
        abc = HEAD + "C2 D2 E2 F2 G2 | C8 |]\n"
        with pytest.raises(ABCValidationError, match="bar 1") as caught:
            validate_abc(abc)
        assert caught.value.code is ABCErrorCode.BAR_LENGTH

    def test_underfull_interior_bar_rejected(self):
        abc = HEAD + "C8 | D4 | E8 | C8 |]\n"
        with pytest.raises(ABCValidationError, match="bar 2"):
            validate_abc(abc)

    def test_pickup_and_final_bars_may_be_short(self):
        validate_abc(HEAD + "GA | C2 E2 G2 c2 | C6 |]\n")

    def test_triplets_count_correctly(self):
        validate_abc(HEAD + "(3CDE F2 G2 A2 | C8 |]\n")

    def test_chords_and_chord_symbols(self):
        validate_abc(HEAD + '"C"[CEG]2 [DF]2 "G"G2 B2 | C8 |]\n')

    def test_empty_chord_rejected(self):
        with pytest.raises(ABCValidationError, match="no notes inside") as caught:
            validate_abc(HEAD + "[]8 | C8 |]\n")
        assert caught.value.code is ABCErrorCode.EMPTY_CHORD

    def test_rests_and_multimeasure_rests(self):
        validate_abc(HEAD + "z2 C2 z4 | Z | C8 |]\n")

    def test_bad_meter_header(self):
        with pytest.raises(ABCValidationError, match="not a meter") as caught:
            validate_abc("X:1\nT:x\nM:waltz\nL:1/8\nK:C\nC8 |]\n")
        assert caught.value.code is ABCErrorCode.BAD_METER

    def test_bad_unit_length_header(self):
        with pytest.raises(ABCValidationError, match="unit note length") as caught:
            validate_abc("X:1\nT:x\nM:4/4\nL:quaver\nK:C\nC8 |]\n")
        assert caught.value.code is ABCErrorCode.BAD_UNIT_LENGTH

    def test_six_eight_meter(self):
        validate_abc("X:1\nT:x\nM:6/8\nL:1/8\nK:G\nGAB dBG | G3 g3 |]\n")

    def test_free_meter_skips_duration_checks(self):
        validate_abc("X:1\nT:x\nM:none\nL:1/8\nK:C\nC3 | D | EFG |]\n")

    def test_additive_meter(self):
        validate_abc("X:1\nT:x\nM:(2+3+2)/8\nL:1/8\nK:C\nCD EFG AB | C7 |]\n")

    def test_line_without_a_barline_runs_on(self):
        validate_abc(HEAD + "CDEF\nGABc | cdefgabc |]\n")

    def test_trailing_comment_ignored(self):
        validate_abc(HEAD + "CDEFGABc | % the first phrase\ncdefgabc |]\n")


class TestBarlines:
    """Every ABC 2.1 barline has to tokenize whole (finding: false rejects)."""

    def test_bracket_repeat_start(self):
        validate_abc(HEAD + "[|:CDEFGABc | cdefgabc |]\n")

    def test_repeat_end_with_thin_thick(self):
        validate_abc(HEAD + "|:CDEFGABc | cdefgabc :|]\n")

    def test_double_colon_barline(self):
        validate_abc(HEAD + "|: C8 | C8 :: C8 | C8 :|\n")

    def test_double_barline_and_final_barline(self):
        validate_abc(HEAD + "C8 || C8 | C8 |]\n")

    @pytest.mark.parametrize(
        ("line", "count"),
        [
            ("[|:C8 | C8 :|]", 2),
            ("|:C8 | C8 ::C8 | C8:|", 4),
            ("C8 |[1 C8 :|[2 C8 |]", 3),
        ],
    )
    def test_barlines_do_not_leak_into_bars(self, line: str, count: int):
        bars = voice_bars(HEAD + line + "\n")["1"]
        assert len(bars) == count, bars
        # only the volta number may survive; no barline glyph ever does.
        assert all(not set(bar) & set("|:]") for bar in bars), bars


class TestSectionEdges:
    """A repeat sign never buys a bar a free pass (finding: escape hatches)."""

    def test_repeat_does_not_exempt_interior_bars(self):
        abc = HEAD + "|:C4 | D4 | E4 | F4 | G4 :|\n"
        with pytest.raises(ABCValidationError, match="bar 2") as caught:
            validate_abc(abc)
        assert caught.value.code is ABCErrorCode.BAR_LENGTH

    def test_repeat_does_not_exempt_a_single_short_middle_bar(self):
        abc = HEAD + "|: CDEF GABc | CD | CDEF GABc :|\n"
        with pytest.raises(ABCValidationError, match="bar 2"):
            validate_abc(abc)

    def test_two_bar_tune_that_does_not_add_up_is_rejected(self):
        # 2/8 + 2/8 is not a bar between them: this is two short bars, not
        # a pickup, and a two-bar tune must be checked like any other.
        with pytest.raises(ABCValidationError, match="bar 1") as caught:
            validate_abc(HEAD + "CD | EF |\n")
        assert caught.value.code is ABCErrorCode.BAR_LENGTH

    def test_two_bar_tune_may_not_be_long(self):
        with pytest.raises(ABCValidationError, match="more than the meter"):
            validate_abc(HEAD + "CDEFGABcd | EF |\n")

    def test_no_bar_may_exceed_the_meter(self):
        with pytest.raises(ABCValidationError, match="more than the meter"):
            validate_abc(HEAD + "C8 | C8 | CDEFGABcd |]\n")

    def test_bar_before_a_repeat_end_may_be_short(self):
        validate_abc(HEAD + "C8 | C8 | C4 :| C8 | C8 |]\n")

    def test_bar_after_a_repeat_start_may_be_short(self):
        validate_abc(HEAD + "C8 | C8 |: C4 | C8 | C8 |]\n")

    def test_volta_bar_may_be_short(self):
        validate_abc(HEAD + "|: C8 | C8 |1 C4 :|2 C8 |]\n")


class TestAnacrusis:
    """A pickup is only a pickup if the final bar completes it."""

    def test_pickup_and_final_bar_summing_to_one_bar_accepted(self):
        validate_abc(HEAD + "C | CDEFGABc | CDEFGABc | CDEFGAB |]\n")

    def test_pickup_and_final_bar_that_do_not_sum_rejected(self):
        # 1/8 pickup with a 1/8 final bar: seven eighths go missing.
        abc = HEAD + "C | CDEFGABc | CDEFGABc | CDEFGABc | C |]\n"
        with pytest.raises(ABCValidationError, match="bar 1") as caught:
            validate_abc(abc)
        assert caught.value.code is ABCErrorCode.BAR_LENGTH

    def test_short_final_bar_without_a_pickup_rejected(self):
        with pytest.raises(ABCValidationError, match="bar 3") as caught:
            validate_abc(HEAD + "CDEFGABc | CDEFGABc | C4 |]\n")
        assert caught.value.code is ABCErrorCode.BAR_LENGTH

    def test_scarborough_style_pickup_in_three_four(self):
        # 2/8 pickup answered by a 4/8 final bar, in 3/4: they make a bar.
        validate_abc("X:1\nT:x\nM:3/4\nL:1/8\nK:Ddor\nA2 | d3 e d2 | D4 |]\n")

    def test_a_lone_short_bar_is_not_a_pickup(self):
        with pytest.raises(ABCValidationError, match="bar 1"):
            validate_abc(HEAD + "CD |]\n")


class TestVoiceTotals:
    """Bar counts agreeing is not the same as voices staying together."""

    def test_a_pickup_in_only_one_voice_is_rejected(self):
        # Three bars each, so the counts agree — but voice 1 carries a
        # pickup voice 2 never answers, so they play a quarter apart.
        abc = (
            "X:1\nT:duet\nM:4/4\nL:1/8\n"
            'V:1 name="flute"\nV:2 name="cello"\nK:C\n'
            "V:1\nCD | CDEFGABc | CDEF GA |]\n"
            "V:2\nC,8 | C,8 | C,8 |]\n"
        )
        with pytest.raises(ABCValidationError, match="different amounts") as caught:
            validate_abc(abc)
        assert caught.value.code is ABCErrorCode.VOICE_MISALIGNED

    def test_the_desynced_first_bar_case_is_rejected(self):
        # The reported hole: voice 1's first bar is 1/8 against voice 2's
        # 8/8, and the bar counts match, so nothing used to catch it.
        abc = (
            "X:1\nT:duet\nM:4/4\nL:1/8\n"
            'V:1 name="flute"\nV:2 name="cello"\nK:C\n'
            "V:1\nC | CDEFGABc | CDEFGABc |]\n"
            "V:2\nCDEFGABc | CDEFGABc | CDEFGABc |]\n"
        )
        with pytest.raises(ABCValidationError):
            validate_abc(abc)

    def test_voices_with_matching_pickups_are_accepted(self):
        abc = (
            "X:1\nT:duet\nM:4/4\nL:1/8\n"
            'V:1 name="flute"\nV:2 name="cello"\nK:C\n'
            "V:1\nCD | CDEFGABc | CDEF GA |]\n"
            "V:2\nz2 | C,8 | C,6 |]\n"
        )
        validate_abc(abc)


class TestMultiMeasureRests:
    """Z<n>/X<n> count as n whole bars (finding: false reject + misalignment)."""

    def test_multimeasure_rest_counts_as_two_bars(self):
        validate_abc(HEAD + "C8 | Z2 | C8 |]\n")

    def test_invisible_multimeasure_rest(self):
        validate_abc(HEAD + "C8 | X3 | C8 |]\n")

    def test_multimeasure_rest_keeps_voices_aligned(self):
        abc = (
            "X:1\nT:duet\nM:4/4\nL:1/8\n"
            'V:1 name="flute"\nV:2 name="cello"\nK:C\n'
            "V:1\nC8 | C8 | C8 | C8 | C8 | C8 | C8 | C8 |]\n"
            "V:2\nZ2 | C,8 | C,8 | C,8 | C,8 | C,8 | C,8 |]\n"
        )
        validate_abc(abc)

    def test_multimeasure_rest_of_the_wrong_length_misaligns(self):
        abc = (
            "X:1\nT:duet\nM:4/4\nL:1/8\n"
            'V:1 name="flute"\nV:2 name="cello"\nK:C\n'
            "V:1\nC8 | C8 | C8 | C8 |]\n"
            "V:2\nZ2 | C,8 |]\n"
        )
        with pytest.raises(ABCValidationError, match="different lengths") as caught:
            validate_abc(abc)
        assert caught.value.code is ABCErrorCode.VOICE_MISALIGNED

    def test_notes_beside_a_multimeasure_rest_are_still_counted(self):
        with pytest.raises(ABCValidationError, match="more than the meter"):
            validate_abc(HEAD + "C8 | Z2 C2 | C8 |]\n")


class TestInlineFields:
    """Inline [M:]/[L:] are applied; everything else is duration-neutral."""

    def test_inline_meter_change_is_applied(self):
        validate_abc(HEAD + "C8 | [M:3/4]C6 | C6 |]\n")

    def test_inline_meter_change_is_enforced(self):
        with pytest.raises(ABCValidationError, match="3/4") as caught:
            validate_abc(HEAD + "C8 | [M:3/4]C8 | C6 |]\n")
        assert caught.value.code is ABCErrorCode.BAR_LENGTH

    def test_inline_unit_length_change_is_applied(self):
        validate_abc(HEAD + "C8 | [L:1/4]C4 | C4 |]\n")

    def test_neutral_inline_field_keeps_the_bar_verifiable(self):
        with pytest.raises(ABCValidationError, match="bar 2"):
            validate_abc(HEAD + "CDEFGABc | [K:G]C | cdefgabc |]\n")

    def test_neutral_inline_field_in_a_full_bar_passes(self):
        validate_abc(HEAD + "CDEFGABc | [K:G]cdefgabc | cdefgabc |]\n")


class TestTuplets:
    """(5 (7 (9 take the ABC 2.2 meter-dependent default rather than a pass."""

    def test_quintuplet_in_simple_meter_takes_two_units(self):
        validate_abc(HEAD + "(5CDEFG G2A2c2 | C8 |]\n")

    def test_quintuplet_in_compound_meter_takes_three_units(self):
        validate_abc("X:1\nT:x\nM:6/8\nL:1/8\nK:G\nG3 g3 | (5CDEFG def |]\n")

    def test_septuplet_counted(self):
        validate_abc(HEAD + "(7CDEFGAB CDEFGA | C8 |]\n")

    def test_short_bar_with_a_quintuplet_is_still_caught(self):
        with pytest.raises(ABCValidationError, match="bar 2") as caught:
            validate_abc(HEAD + "CDEFGABc | (5CDEFG C | CDEFGABc |]\n")
        assert caught.value.code is ABCErrorCode.BAR_LENGTH

    def test_explicit_tuplet_ratio_still_wins(self):
        validate_abc(HEAD + "(5:4:5CDEFG C4 | C8 |]\n")


class TestHeaderSemantics:
    """Header and directive values have to mean something (minor finding)."""

    @pytest.mark.parametrize(
        "key", ["C", "Ddor", "F#m", "Bb", "Amix", "Emin", "Gmajor", "none", "HP"]
    )
    def test_plausible_keys_accepted(self, key: str):
        validate_abc(f"X:1\nT:x\nM:4/4\nL:1/8\nK:{key}\nC8 |]\n")

    def test_key_may_carry_extra_fields(self):
        validate_abc("X:1\nT:x\nM:4/4\nL:1/8\nK:C clef=bass\nC8 |]\n")

    @pytest.mark.parametrize("key", ["Hqz", "Q", "Cxyz", "42"])
    def test_nonsense_keys_rejected(self, key: str):
        with pytest.raises(ABCValidationError, match="not a key") as caught:
            validate_abc(f"X:1\nT:x\nM:4/4\nL:1/8\nK:{key}\nC8 |]\n")
        assert caught.value.code is ABCErrorCode.BAD_KEY

    @pytest.mark.parametrize(
        "tempo", ["1/4=120", '"Andante" 1/4=72', "96", '"largo"', "1/4 3/8=40"]
    )
    def test_plausible_tempos_accepted(self, tempo: str):
        validate_abc(f"X:1\nT:x\nM:4/4\nL:1/8\nQ:{tempo}\nK:C\nC8 |]\n")

    @pytest.mark.parametrize("tempo", ["not a tempo at all", "1/4=", "fast=120"])
    def test_nonsense_tempos_rejected(self, tempo: str):
        with pytest.raises(ABCValidationError, match="not a tempo") as caught:
            validate_abc(f"X:1\nT:x\nM:4/4\nL:1/8\nQ:{tempo}\nK:C\nC8 |]\n")
        assert caught.value.code is ABCErrorCode.BAD_TEMPO

    def test_two_tunes_rejected(self):
        abc = "X:1\nT:a\nM:4/4\nL:1/8\nK:C\nC8 |]\nX:2\nT:b\nK:C\nD8 |]\n"
        with pytest.raises(ABCValidationError, match="exactly one tune") as caught:
            validate_abc(abc)
        assert caught.value.code is ABCErrorCode.MULTIPLE_TUNES

    @pytest.mark.parametrize("note", ["C0", "z0", "[CEG]0"])
    def test_zero_duration_rejected(self, note: str):
        with pytest.raises(ABCValidationError, match="length of zero") as caught:
            validate_abc(HEAD + f"{note} C8 | C8 |]\n")
        assert caught.value.code is ABCErrorCode.BAD_DURATION

    @pytest.mark.parametrize("program", [0, 42, 127])
    def test_midi_programs_in_range_accepted(self, program: int):
        abc = f"X:1\nT:x\nM:4/4\nL:1/8\n%%MIDI program {program}\nK:C\nC8 |]\n"
        validate_abc(abc)

    @pytest.mark.parametrize("program", [999, 128, -1])
    def test_midi_programs_out_of_range_rejected(self, program: int):
        abc = f"X:1\nT:x\nM:4/4\nL:1/8\n%%MIDI program {program}\nK:C\nC8 |]\n"
        with pytest.raises(ABCValidationError, match="0 to 127") as caught:
            validate_abc(abc)
        assert caught.value.code is ABCErrorCode.BAD_MIDI_PROGRAM


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

    def test_unequal_complementary_pickups_rejected(self):
        # Each voice's pickup completes its own final bar, so bar counts
        # and totals agree — but the voices enter an eighth apart.
        bad = (
            "X:1\nT:x\nM:4/4\nL:1/8\nK:C\n"
            "V:1\nC | CDEFGABc | CDEFGAB |\n"
            "V:2\nC,D, | C,8 | C,6 |\n"
        )
        with pytest.raises(ABCValidationError, match="different pickups"):
            validate_abc(bad)

    def test_matching_pickups_accepted(self):
        good = (
            "X:1\nT:x\nM:4/4\nL:1/8\nK:C\n"
            "V:1\nC | CDEFGABc | CDEFGAB |\n"
            "V:2\nC, | C,8 | C,4C,2C, |\n"
        )
        validate_abc(good)


def test_dynamics_and_articulations_validate():
    abc = HEAD + "!p!.C2 D2 ~E2 uF2 | !f!(GA) (3cde f2 c2 | C8 |]\n"
    validate_abc(abc)


def test_grace_notes_and_ties_are_duration_neutral():
    validate_abc(HEAD + "{gag}C2 D2- D2 E2 | C8 |]\n")


def test_linebreak_marker_is_duration_neutral():
    """``$`` is the ABC 2.1 linebreak marker; it holds no time."""
    validate_abc("X:1\nT:x\nM:4/4\nL:1/8\nI:linebreak $\nK:C\nCDEF$GABc | C8 |]\n")


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_folk_tune_fixtures_validate(path: Path):
    """A strict parser earns its keep only if real tunes pass it."""
    validate_abc(path.read_text(encoding="utf-8"))


def test_fixtures_are_actually_present():
    assert len(FIXTURES) >= 4


def test_fixture_voices_stay_aligned_through_a_multimeasure_rest():
    source = (Path(__file__).parent / "fixtures" / "evening-round.abc").read_text()
    bars = voice_bars(source)
    # voice 3 writes 7 segments but fills 8 bars, thanks to its Z2.
    assert len(bars["1"]) == 8
    assert len(bars["3"]) == 7
    validate_abc(source)
