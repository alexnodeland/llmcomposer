import asyncio
import re

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from llmcomposer.agent import (
    COMPOSER_INSTRUCTIONS,
    PROMPT_SHA,
    PROMPT_VERSION,
    CompositionDeps,
    _midi_to_abc,
    composer_agent,
)
from llmcomposer.descriptors import GM_RANGES

# The digest of the prompt the study currently runs. It is pinned so an
# edit to COMPOSER_INSTRUCTIONS — including one arriving through GM_RANGES,
# since the range table is generated from it — fails here until PROMPT_VERSION
# and this constant are bumped together, deliberately.
EXPECTED_PROMPT_SHA = "1bdbb33733949d22"

# Which General MIDI program each instrument the prompt names is scored
# against. Written out here rather than imported so a change to the pairing
# in agent.py has to be made twice, on purpose.
PROMPT_PROGRAMS = {
    "flute": 73,
    "oboe": 68,
    "clarinet": 71,
    "bassoon": 70,
    "horn": 60,
    "trumpet": 56,
    "trombone": 57,
    "tuba": 58,
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "harp": 46,
    "piano": 0,
}

_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_TOKEN = re.compile(r"([_^=]*)([A-Ga-g])([,']*)")
_RANGE_ROW = re.compile(r"([a-z]+) +(\S+) +- +(\S+)")


def _midi(token: str) -> int:
    """Read a bare ABC note token as a MIDI pitch, middle C at 60."""
    match = _TOKEN.fullmatch(token)
    assert match is not None, token
    accidental, letter, marks = match.groups()
    octave = (4 if letter.isupper() else 5) + marks.count("'") - marks.count(",")
    shift = accidental.count("^") - accidental.count("_")
    return (octave + 1) * 12 + _SEMITONES[letter.upper()] + shift


def _printed_ranges() -> dict[str, tuple[int, int]]:
    """Read the prompt's instrument-range table back into MIDI numbers."""
    start = COMPOSER_INSTRUCTIONS.index("C, the octave below:\n\n")
    table = COMPOSER_INSTRUCTIONS[start:].split("\n\n")[1]
    rows = {
        name: (_midi(low), _midi(high)) for name, low, high in _RANGE_ROW.findall(table)
    }
    assert rows, table
    return rows


VALID_ABC = "X:1\nT:test tune\nM:4/4\nL:1/8\nQ:1/4=80\nK:C\nCD EF G2 AB | c8 |]\n"

# Bar 2 holds nine eighth notes where the meter asks for eight, and it is
# neither the first nor the last bar — an unambiguous bar-length rejection.
LONG_BAR_ABC = (
    "X:1\nT:overfull\nM:4/4\nL:1/8\nQ:1/4=80\nK:C\n"
    "CDEF GABc | cdefgabcd | CDEF GABc | c8 |]\n"
)


def _tool_call(info: AgentInfo, abc: str, reply: str = "ok") -> ModelResponse:
    """Wrap a candidate score in the agent's structured-output tool call."""
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name=info.output_tools[0].name,
                args={"reply": reply, "abc": abc},
            )
        ]
    )


def _saw_retry(messages: list[ModelMessage]) -> bool:
    """Return whether the validator has already bounced a score this run."""
    return any(
        isinstance(part, RetryPromptPart)
        for message in messages
        for part in message.parts
    )


def test_agent_returns_validated_score_update():
    model = TestModel(custom_output_args={"reply": "here it is", "abc": VALID_ABC})
    deps = CompositionDeps()
    result = asyncio.run(composer_agent.run("something gentle", deps=deps, model=model))
    assert result.output.reply == "here it is"
    assert result.output.abc.startswith("X:1")
    assert deps.bounces == []


def test_invalid_abc_triggers_model_retry():
    attempts: list[bool] = []

    def flaky_composer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # First attempt returns garbage; the retry returns a valid score.
        saw_retry = _saw_retry(messages)
        attempts.append(saw_retry)
        return _tool_call(info, VALID_ABC if saw_retry else "not a score at all")

    result = asyncio.run(
        composer_agent.run(
            "anything", deps=CompositionDeps(), model=FunctionModel(flaky_composer)
        )
    )
    assert attempts == [False, True]
    assert result.output.abc.startswith("X:1")


def test_a_bounced_score_is_recorded_with_its_code_and_the_rejected_abc():
    def malformer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Attempt 1 overfills bar 2; attempt 2 returns a well-formed tune.
        return _tool_call(info, VALID_ABC if _saw_retry(messages) else LONG_BAR_ABC)

    deps = CompositionDeps()
    result = asyncio.run(
        composer_agent.run("anything", deps=deps, model=FunctionModel(malformer))
    )
    assert result.output.abc.startswith("X:1")
    assert len(deps.bounces) == 1
    bounce = deps.bounces[0]
    assert bounce.attempt == 1
    assert bounce.code == "bar_length"
    assert bounce.rejected_abc == LONG_BAR_ABC.strip()
    assert "bar 2" in bounce.message


def test_current_score_is_injected_into_instructions():
    captured: list[str] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured.append(messages[-1].instructions or "")
        return _tool_call(info, VALID_ABC)

    deps = CompositionDeps(current_abc=VALID_ABC)
    asyncio.run(composer_agent.run("brighter", deps=deps, model=FunctionModel(capture)))
    assert "<current_score>" in captured[0]
    assert "T:test tune" in captured[0]


def test_instructions_teach_form_cadence_voice_leading_and_scoring():
    prompt = COMPOSER_INSTRUCTIONS
    for section in ("# Form", "# Harmony and cadence", "# Voice-leading"):
        assert section in prompt
    for section in ("# Scoring and registers", "# Notation", "# Reply"):
        assert section in prompt
    for craft in ("|:", ":|", "[1", "AABB", "antecedent", "half cadence"):
        assert craft in prompt
    for scoring in ("clef=bass", "clef=alto", 'sname="fl."', "%%score [1 2]", "K:Ddor"):
        assert scoring in prompt
    assert "never exceed" in prompt.lower()
    assert PROMPT_VERSION == "composer-v3"


def test_the_prompt_digest_is_pinned_to_its_declared_version():
    # Bump EXPECTED_PROMPT_SHA and PROMPT_VERSION together, never one alone.
    assert PROMPT_SHA == EXPECTED_PROMPT_SHA
    assert len(PROMPT_SHA) == 16


def test_midi_to_abc_spells_the_octaves_the_way_the_parser_reads_them():
    assert _midi_to_abc(60) == "C"
    assert _midi_to_abc(72) == "c"
    assert _midi_to_abc(96) == "c''"
    assert _midi_to_abc(48) == "C,"
    assert _midi_to_abc(21) == "A,,,,"
    assert _midi_to_abc(58) == "_B,"
    assert _midi_to_abc(54) == "^F,"
    for note in range(21, 109):
        assert _midi(_midi_to_abc(note)) == note


def test_the_printed_ranges_are_the_ranges_the_evals_score_against():
    printed = _printed_ranges()
    assert set(printed) == set(PROMPT_PROGRAMS)
    for name, (low, high) in printed.items():
        program = PROMPT_PROGRAMS[name]
        assert program in GM_RANGES, name
        gm = GM_RANGES[program]
        assert (low, high) == (gm.low, gm.high), f"{name} drifted from {gm.name}"


def test_every_general_midi_patch_the_prompt_offers_is_a_known_program():
    start = COMPOSER_INSTRUCTIONS.index("%%MIDI program <0-127>")
    bullet = COMPOSER_INSTRUCTIONS[start:].split("\n-")[0]
    offered = re.findall(r"(\d+) ([a-z ]+?)[,\.\n]", bullet)
    assert len(offered) >= 10
    for program, label in offered:
        gm = GM_RANGES.get(int(program))
        assert gm is not None, label
        assert label.split()[-1] in gm.name, f"{label} is not {gm.name}"


def test_the_prompt_teaches_the_dialect_the_validator_actually_accepts():
    flowed = " ".join(COMPOSER_INSTRUCTIONS.split())
    assert "Never use the & voice overlay or $ line-break markers" in flowed
    assert "give every part its own V: line" in flowed


def test_the_score_directive_comes_before_the_voice_declarations():
    prompt = COMPOSER_INSTRUCTIONS
    flowed = " ".join(prompt.split())
    assert "Immediately before the V: declarations write one %%score line" in flowed
    assert "then %%score, then the V: declarations, then K:" in flowed
    example = prompt[prompt.index("X:1\nT:two lamps") : prompt.index("\n\n# Reply")]
    assert example.index("%%score [1 2]") < example.index('V:1 name="flute"')


def test_the_clef_bullet_gives_a_register_rule_not_only_a_list():
    flowed = " ".join(COMPOSER_INSTRUCTIONS.split())
    assert "choose the clef from the register you actually write in" in flowed
    assert "ledger lines" in flowed
    assert "clef=bass for cello" in flowed  # the defaults are still there


def test_the_transposing_instruments_are_named_as_concert_pitch():
    flowed = " ".join(COMPOSER_INSTRUCTIONS.split())
    assert "Clarinet, trumpet and horn" in flowed
    assert "these bounds are concert pitch" in flowed


def test_the_worked_example_in_the_prompt_is_itself_valid_abc():
    from llmcomposer.abc_notation import validate_abc

    start = COMPOSER_INSTRUCTIONS.index("X:1\nT:two lamps")
    end = COMPOSER_INSTRUCTIONS.index("\n\n# Reply")
    validate_abc(COMPOSER_INSTRUCTIONS[start:end])


def test_the_prompt_does_not_contradict_itself_about_length():
    flowed = " ".join(COMPOSER_INSTRUCTIONS.split())
    assert "8 to 32 bars of written music overall" in flowed
    assert "A fresh tune should be 8 to 16 bars" in flowed
    assert "8-16" not in flowed
    assert "8-32" not in flowed
