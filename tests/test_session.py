import asyncio

from llmcomposer.abc_notation import validate_abc
from llmcomposer.session import ComposerSession


def test_offline_session_composes_and_accumulates_history():
    session = ComposerSession(model="offline")
    update, meta = asyncio.run(session.send("like rain on a window"))
    validate_abc(update.abc)
    assert meta.requests >= 1
    assert meta.corrections == 0
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
    async def collect():
        session = ComposerSession(model="offline")
        events = []
        async for event in session.send_stream("like rain on a window"):
            events.append(event)
        return session, events

    session, events = asyncio.run(collect())
    types = [event["type"] for event in events]
    assert types[0] == "writing"
    assert "progress" in types
    assert types[-1] == "final"
    final = events[-1]
    validate_abc(final["abc"])
    assert session.abc == final["abc"]
    assert final["meta"]["requests"] >= 1
