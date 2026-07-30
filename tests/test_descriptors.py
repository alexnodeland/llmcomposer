"""Hand-computed cases for the symbolic descriptors."""

import math
from pathlib import Path

from llmcomposer.agent import COMPOSER_INSTRUCTIONS
from llmcomposer.descriptors import GM_RANGES, describe, key_signature

C_MAJOR_SCALE = """X:1
T:scale
M:4/4
L:1/8
K:C
CDEF GABc | cBAG FEDC | CDEF GABc | c8 |]
"""

PARALLEL_OCTAVES = """X:1
T:parallels
M:4/4
L:1/8
V:1 name="flute"
%%MIDI program 73
V:2 name="cello"
%%MIDI program 42
K:C
V:1
c4 d4 | e4 f4 |
V:2
C4 D4 | E4 F4 |
"""

FIRST_AND_SECOND_ENDINGS = """X:1
T:two lamps
M:4/4
L:1/8
V:1 name="flute"
%%MIDI program 73
V:2 name="cello"
%%MIDI program 42
K:Ddor
V:1
|: d2 A2 F2 A2 | G2 E2 C2 E2 | F2 A2 d2 c2 |1 A6 z2 :|2 d6 z2 |]
V:2
|: D,4 A,,4 | C,4 G,,4 | D,4 F,4 |1 A,,6 z2 :|2 D,8 |]
"""

FLUTE_IN_THE_CELLAR = """X:1
T:too low to blow
M:4/4
L:1/8
V:1 name="flute"
%%MIDI program 73
K:C
C,8 | D,8 |]
"""


def _tune(key: str, body: str, meter: str = "4/4") -> str:
    return f"X:1\nT:t\nM:{meter}\nL:1/8\nK:{key}\n{body}\n"


# --- key signatures --------------------------------------------------------


def test_key_signature_naturals_and_sharps():
    assert key_signature("C") == {}
    assert key_signature("G") == {"F": 1}
    assert key_signature("D") == {"F": 1, "C": 1}
    assert key_signature("Bb") == {"B": -1, "E": -1}


def test_modal_key_signatures_follow_their_parent_major():
    assert key_signature("Ddor") == {}
    assert key_signature("Emin") == {"F": 1}
    assert key_signature("Gmix") == {}
    assert key_signature("Amix") == {"F": 1, "C": 1}


# --- pitch content ---------------------------------------------------------


def test_c_major_scale_infers_c_with_high_correlation():
    described = describe(C_MAJOR_SCALE)
    assert described.inferred_key == "C"
    assert described.key_match is True
    assert described.key_correlation > 0.85
    assert described.declared_mode == "major"


def test_key_signature_is_applied_when_decoding_pitches():
    # In G major every written F sounds as F#, so the pitch-class set is
    # the G-major scale and the inferred key is G, not C.
    described = describe(_tune("G", "GABc defg | g8 |]"))
    assert described.inferred_key == "G"
    assert described.voice_ranges["1"] == (67, 79)


def test_accidental_persists_for_the_rest_of_its_bar():
    described = describe(_tune("C", "^FF F2 F4 | F8 |]"))
    # Four F#s in bar one (66), then a natural F in bar two (65).
    assert described.voice_ranges["1"] == (65, 66)


def test_dorian_is_measured_by_its_natural_sixth():
    # The same melody twice, differing in one pitch class: the true dorian
    # keeps its natural B, the collapse to D aeolian flattens it.
    dorian = describe(_tune("Ddor", "DEFG ABcd | d8 |]"))
    aeolian = describe(_tune("Ddor", "DEFG A_Bcd | d8 |]"))
    assert dorian.mode_match is True
    assert aeolian.mode_match is False
    assert dorian.diatonic_adherence is not None
    assert aeolian.diatonic_adherence is not None
    assert dorian.diatonic_adherence == 1.0
    assert dorian.diatonic_adherence > aeolian.diatonic_adherence


def test_mixolydian_is_measured_by_its_flat_seventh():
    mixolydian = describe(_tune("Amix", "ABcd efga | a8 |]"))
    major = describe(_tune("Amix", "ABcd ef^ga | a8 |]"))
    assert mixolydian.mode_match is True
    assert major.mode_match is False
    assert mixolydian.diatonic_adherence is not None
    assert major.diatonic_adherence is not None
    assert mixolydian.diatonic_adherence > major.diatonic_adherence


def test_a_tune_outside_its_declared_mode_loses_adherence():
    # Five of the eight pitch classes are foreign to C major.
    chromatic = describe(_tune("C", "C2^C2 D2^D2 | E2^F2 ^G2^A2 |]"))
    assert chromatic.diatonic_adherence == 0.375


def test_a_single_repeated_pitch_has_zero_entropy():
    described = describe(_tune("C", "C8 | C8 |]"))
    assert described.pitch_class_entropy == 0.0


def test_repetition_rate_counts_recurring_three_grams():
    described = describe(_tune("C", "CDEF CDEF | CDEF CDEF |]"))
    # Every 3-gram after the first bar's four has been seen before.
    assert described.ngram_repetition_rate > 0.5
    assert describe(_tune("C", "CDEG AcBd | c8 |]")).ngram_repetition_rate == 0.0


def test_intervals_and_leaps():
    described = describe(_tune("C", "CGCG CGCG | C8 |]"))
    assert math.isclose(described.mean_abs_interval, 7.0, rel_tol=1e-6)
    assert described.leap_rate == 1.0


def test_rest_fraction_is_duration_weighted():
    described = describe(_tune("C", "C4 z4 | C4 z4 |]"))
    assert math.isclose(described.rest_fraction, 0.5, rel_tol=1e-9)


# --- register and voices ---------------------------------------------------


def test_flute_written_below_its_range_is_a_violation():
    described = describe(FLUTE_IN_THE_CELLAR)
    assert described.voice_ranges["1"] == (48, 50)
    assert len(described.range_violations) == 1
    assert "below the instrument's range" in described.range_violations[0]
    assert "flute" in described.range_violations[0]


def test_a_voice_inside_its_range_reports_no_violation():
    described = describe(FLUTE_IN_THE_CELLAR.replace("C,8 | D,8", "c8 | d8"))
    assert described.range_violations == []
    assert described.voice_programs == {"1": 73}


def test_gm_range_table_covers_the_patches_the_agent_is_told_to_use():
    for program in (0, 11, 24, 32, 40, 42, 46, 56, 68, 71, 73):
        assert program in GM_RANGES
    assert GM_RANGES[73].name == "flute"


def test_the_cello_reaches_thumb_position():
    # A cello plays well above E5; stopping the table at 76 made a whole
    # register of ordinary writing read as a range violation.
    assert GM_RANGES[42].high >= 81
    cello = (
        'X:1\nT:high air\nM:4/4\nL:1/8\nV:1 name="cello"\n'
        "%%MIDI program 42\nK:C\na4 a4 | a8 |]\n"
    )
    described = describe(cello)
    assert described.voice_ranges["1"] == (81, 81)
    assert described.range_violations == []


def test_parallel_octaves_between_outer_voices():
    described = describe(PARALLEL_OCTAVES)
    assert described.parallel_octave_count == 3
    assert described.parallel_fifth_count == 0
    assert described.voice_count == 2


def test_parallel_fifths_between_outer_voices():
    # c/F, d/G, e/A, f/_B are all perfect twelfths, moving up together.
    fifths = PARALLEL_OCTAVES.replace("C4 D4 | E4 F4", "F,4 G,4 | A,4 _B,4")
    described = describe(fifths)
    assert described.parallel_fifth_count == 3
    assert described.parallel_octave_count == 0


def test_contrary_motion_is_not_parallel():
    contrary = PARALLEL_OCTAVES.replace("C4 D4 | E4 F4", "C4 B,4 | A,4 G,4")
    assert describe(contrary).parallel_octave_count == 0


def test_a_first_and_second_ending_do_not_sound_back_to_back():
    # The two endings are written side by side but never heard that way:
    # the repeat sits between them, so this is not a parallel octave.
    described = describe(FIRST_AND_SECOND_ENDINGS)
    assert described.parallel_octave_count == 0
    assert described.parallel_fifth_count == 0


def test_the_prompts_own_worked_example_has_no_phantom_parallels():
    """The exemplar the model is told to copy must survive its own analyzer."""
    tail = COMPOSER_INSTRUCTIONS[COMPOSER_INSTRUCTIONS.rindex("\nX:1\n") :]
    heading = tail.find("\n# ")
    described = describe((tail if heading < 0 else tail[:heading]).strip())
    assert described.voice_count == 2
    assert described.parallel_octave_count == 0
    assert described.parallel_fifth_count == 0


def test_parallels_inside_a_volta_tune_are_still_caught():
    # Same tune, with the octaves moved into the body of a strain where
    # they really do sound one after the other.
    caught = FIRST_AND_SECOND_ENDINGS.replace(
        "|: d2 A2 F2 A2 |", "|: d2 e2 f2 g2 |"
    ).replace("|: D,4 A,,4 |", "|: D,2 E,2 F,2 G,2 |")
    assert describe(caught).parallel_octave_count == 3


def test_voice_crossing_is_counted_when_the_lower_voice_climbs_over():
    crossed = PARALLEL_OCTAVES.replace("C4 D4 | E4 F4", "c'4 d'4 | e'4 f'4")
    assert describe(crossed).voice_crossing_count == 4


# --- cadence ---------------------------------------------------------------


def test_authentic_cadence_on_a_v_i_close():
    assert describe(_tune("C", "CDEF GABc | G,8 | C,8 |]")).cadence == "authentic"


def test_plagal_and_half_and_deceptive_closes():
    assert describe(_tune("C", "CDEF GABc | F,8 | C,8 |]")).cadence == "plagal"
    assert describe(_tune("C", "CDEF GABc | C,8 | G,8 |]")).cadence == "half"
    assert describe(_tune("C", "CDEF GABc | G,8 | A,8 |]")).cadence == "deceptive"


def test_an_unresolved_ending_has_no_cadence():
    assert describe(_tune("C", "CDEF GABc | D,8 | E,8 |]")).cadence == "none"


# --- shape of the row itself ----------------------------------------------


def test_counts_and_meter_are_reported():
    described = describe(_tune("Ddor", "DEFG ABcd | d8 |]", meter="7/8"))
    assert described.meter == "7/8"
    assert described.declared_key == "Ddor"
    assert described.declared_mode == "dorian"
    assert described.bar_count == 2
    assert described.note_count == 9
    assert described.voice_count == 1


def test_a_mode_never_contradicted_is_undecided_not_failed():
    # E dorian whose characteristic degree (C#/C) never sounds at all: a
    # hexatonic jig that contradicts nothing is not a mode failure.
    jig = (Path(__file__).parent / "fixtures" / "blackthorn-jig.abc").read_text()
    described = describe(jig)
    assert described.mode_match is None
    assert described.diatonic_adherence == 1.0


def test_malformed_input_still_yields_a_row():
    for junk in ("", "   ", "not music at all", "X:1\nM:4/4\n"):
        described = describe(junk)
        assert described.note_count == 0
        assert described.cadence == "none"
        assert described.range_violations == []
        # nothing sounded, so the mode is neither kept nor broken.
        assert described.diatonic_adherence is None
        assert described.mode_match is None
