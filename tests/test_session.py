import asyncio
import json

import pytest
from pydantic_ai.messages import ModelMessage, RetryPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from llmcomposer.abc_notation import validate_abc
from llmcomposer.agent import PROMPT_SHA, PROMPT_VERSION
from llmcomposer.recording import RECORD_DIR_ENV
from llmcomposer.session import ComposerSession, resolve_sampling

VALID_ABC = "X:1\nT:test tune\nM:4/4\nL:1/8\nQ:1/4=80\nK:C\nCD EF G2 AB | c8 |]\n"

# Bar 2 holds nine eighths where the meter asks for eight — a bar the
# validator must refuse, and neither the first nor the last bar.
LONG_BAR_ABC = (
    "X:1\nT:overfull\nM:4/4\nL:1/8\nQ:1/4=80\nK:C\n"
    "CDEF GABc | cdefgabcd | CDEF GABc | c8 |]\n"
)


def collect(agen) -> list[dict]:
    async def drain() -> list[dict]:
        return [event async for event in agen]

    return asyncio.run(drain())


def flaky_stream_model() -> FunctionModel:
    """A model that writes an overfull bar first, then a well-formed tune."""

    async def stream(messages: list[ModelMessage], info: AgentInfo):
        bounced = any(
            isinstance(part, RetryPromptPart)
            for message in messages
            for part in message.parts
        )
        abc = VALID_ABC if bounced else LONG_BAR_ABC
        args = json.dumps({"reply": "here", "abc": abc})
        yield {0: DeltaToolCall(name=info.output_tools[0].name)}
        for start in range(0, len(args), 48):
            yield {0: DeltaToolCall(json_args=args[start : start + 48])}

    return FunctionModel(stream_function=stream, model_name="flaky")


def boom_model() -> FunctionModel:
    """A model whose run fails outright, the way a provider outage does."""

    async def stream(messages: list[ModelMessage], info: AgentInfo):
        raise RuntimeError("the provider went quiet")
        yield {}  # pragma: no cover - unreachable, makes this a generator

    return FunctionModel(stream_function=stream, model_name="boom")


def test_offline_session_composes_and_accumulates_history():
    session = ComposerSession(model="offline")
    update, meta = asyncio.run(session.send("like rain on a window"))
    validate_abc(update.abc)
    assert meta.requests >= 1
    assert meta.corrections == 0
    assert meta.bounces == []
    assert meta.prompt_version == PROMPT_VERSION
    assert meta.prompt_sha == PROMPT_SHA
    assert "offline-composer" in meta.model
    assert session.abc == update.abc
    assert "K:Am" in update.abc  # rain maps to a minor key
    first_len = len(session.history)
    assert first_len >= 2

    second, _ = asyncio.run(session.send("keep going, a little brighter"))
    validate_abc(second.abc)
    assert len(session.history) > first_len
    # The revision stays in the key the collaboration established.
    assert "K:Am" in second.abc


def test_reset_clears_score_and_history():
    session = ComposerSession(model="offline")
    asyncio.run(session.send("a bright morning"))
    session.reset()
    assert session.abc is None
    assert session.history == []


def test_model_name_label():
    session = ComposerSession(model="offline")
    assert "offline-composer" in session.model_name


def test_available_models_env_parsing(monkeypatch):
    from llmcomposer.session import available_models

    monkeypatch.delenv("LLMCOMPOSER_MODELS", raising=False)
    monkeypatch.setenv("LLMCOMPOSER_MODEL", "anthropic:claude-opus-5")
    assert available_models() == ["anthropic:claude-opus-5", "offline"]

    monkeypatch.setenv("LLMCOMPOSER_MODELS", "litellm:glm-4.6, litellm:gpt-4o ,offline")
    assert available_models() == ["litellm:glm-4.6", "litellm:gpt-4o", "offline"]


def test_build_model_routes_litellm_through_proxy(monkeypatch):
    from llmcomposer.session import build_model

    monkeypatch.setenv("LITELLM_BASE_URL", "http://proxy.test/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    model, label = build_model("litellm:glm-4.6")
    assert label == "litellm:glm-4.6"
    assert model.model_name == "glm-4.6"  # type: ignore[union-attr]
    assert model.system == "litellm"  # type: ignore[union-attr]


def test_set_model_keeps_history_and_score():
    session = ComposerSession(model="offline")
    update, _ = asyncio.run(session.send("like rain on a window"))
    history_len = len(session.history)
    session.set_model("offline")
    assert session.abc == update.abc
    assert len(session.history) == history_len
    revised, _ = asyncio.run(session.send("keep going, a little brighter"))
    validate_abc(revised.abc)


def test_offline_arrangement_has_aligned_voices_and_patches():
    from llmcomposer.abc_notation import voice_bars

    session = ComposerSession(model="offline")
    update, _ = asyncio.run(session.send("arrange this as a trio, together"))
    validate_abc(update.abc)
    voices = voice_bars(update.abc)
    assert len(voices) == 3
    assert len({len(bars) for bars in voices.values()}) == 1
    assert update.abc.count("%%MIDI program") == 3


def test_send_stream_yields_progress_then_final():
    session = ComposerSession(model="offline")
    events = collect(session.send_stream("like rain on a window"))
    types = [event["type"] for event in events]
    assert types[0] == "writing"
    assert "progress" in types
    assert types[-1] == "final"
    final = events[-1]
    validate_abc(final["abc"])
    assert session.abc == final["abc"]
    assert final["meta"]["requests"] >= 1


def test_resolve_sampling_is_absent_until_configured(monkeypatch):
    monkeypatch.delenv("LLMCOMPOSER_TEMPERATURE", raising=False)
    monkeypatch.delenv("LLMCOMPOSER_SEED", raising=False)
    assert resolve_sampling() == (None, None, None)

    monkeypatch.setenv("LLMCOMPOSER_TEMPERATURE", "0.35")
    monkeypatch.setenv("LLMCOMPOSER_SEED", "1337")
    settings, temperature, seed = resolve_sampling()
    assert settings == {"temperature": 0.35, "seed": 1337}
    assert temperature == 0.35
    assert seed == 1337


def test_resolve_sampling_ignores_unparseable_values(monkeypatch):
    monkeypatch.setenv("LLMCOMPOSER_TEMPERATURE", "warm")
    monkeypatch.setenv("LLMCOMPOSER_SEED", "")
    assert resolve_sampling() == (None, None, None)


def test_explicit_sampling_arguments_beat_the_environment(monkeypatch):
    monkeypatch.setenv("LLMCOMPOSER_TEMPERATURE", "0.9")
    monkeypatch.setenv("LLMCOMPOSER_SEED", "1")
    settings, temperature, seed = resolve_sampling(temperature=0.0, seed=42)
    assert settings == {"temperature": 0.0, "seed": 42}
    assert (temperature, seed) == (0.0, 42)

    # each knob falls back on its own
    _, temperature, seed = resolve_sampling(seed=42)
    assert (temperature, seed) == (0.9, 42)


def test_a_session_can_pin_its_own_sampling(monkeypatch):
    monkeypatch.delenv("LLMCOMPOSER_TEMPERATURE", raising=False)
    monkeypatch.delenv("LLMCOMPOSER_SEED", raising=False)
    session = ComposerSession(model="offline", temperature=0.3, seed=99)
    _, meta = asyncio.run(session.send("a bright morning"))
    assert meta.temperature == 0.3
    assert meta.seed == 99

    # unpinned sessions keep deferring to the environment
    monkeypatch.setenv("LLMCOMPOSER_TEMPERATURE", "0.8")
    _, meta = asyncio.run(ComposerSession(model="offline").send("a bright morning"))
    assert meta.temperature == 0.8
    assert meta.seed is None


def test_sampling_settings_are_recorded_on_the_turn(monkeypatch):
    monkeypatch.setenv("LLMCOMPOSER_TEMPERATURE", "0.2")
    monkeypatch.setenv("LLMCOMPOSER_SEED", "7")
    session = ComposerSession(model="offline")
    _, meta = asyncio.run(session.send("a bright morning"))
    assert meta.temperature == 0.2
    assert meta.seed == 7
    assert meta.prompt_version == PROMPT_VERSION


def test_a_bounce_reaches_the_stream_before_the_rewrite():
    session = ComposerSession(model=flaky_stream_model())
    events = collect(session.send_stream("something in four"))
    types = [event["type"] for event in events]

    writing_at = [index for index, kind in enumerate(types) if kind == "writing"]
    assert len(writing_at) == 2
    # The reason arrives before the model starts writing again.
    assert types.index("bounce") < writing_at[1]

    bounce = next(event for event in events if event["type"] == "bounce")
    assert bounce["attempt"] == 1
    assert bounce["code"]
    assert bounce["reason"]

    final = events[-1]
    assert final["type"] == "final"
    validate_abc(final["abc"])
    meta = final["meta"]
    assert meta["corrections"] == 1
    assert len(meta["bounces"]) == 1
    assert meta["bounces"][0]["code"] == bounce["code"]
    assert meta["bounces"][0]["rejected_abc"] == LONG_BAR_ABC.strip()


def test_a_failing_model_propagates_out_of_the_stream():
    session = ComposerSession(model=boom_model())

    async def drain():
        async for _ in session.send_stream("anything"):
            pass

    with pytest.raises(Exception, match="quiet"):
        asyncio.run(drain())


def test_turns_are_recorded_as_jsonl_when_the_sink_is_configured(monkeypatch, tmp_path):
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    session = ComposerSession(model="offline")
    update, meta = asyncio.run(session.send("like rain on a window"))
    asyncio.run(session.send("keep going, a little brighter"))

    files = sorted(tmp_path.glob("turns-*.jsonl"))
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert [record["turn_index"] for record in records] == [0, 1]
    assert {record["session_id"] for record in records} == {session.session_id}
    assert records[0]["user_message"] == "like rain on a window"
    assert records[0]["reply"] == update.reply
    assert records[0]["abc"] == update.abc
    assert records[0]["prompt_version"] == meta.prompt_version
    assert records[0]["prompt_sha"] == PROMPT_SHA
    assert records[0]["usage"]["requests"] >= 1
    assert records[0]["bounces"] == []


def test_nothing_is_written_when_the_sink_is_not_configured(monkeypatch, tmp_path):
    monkeypatch.delenv(RECORD_DIR_ENV, raising=False)
    session = ComposerSession(model="offline")
    asyncio.run(session.send("a bright morning"))
    assert list(tmp_path.iterdir()) == []
