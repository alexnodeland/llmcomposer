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


def test_offline_arrangement_has_aligned_voices_and_patches():
    from llmcomposer.abc_notation import voice_bars

    session = ComposerSession(model="offline")
    update, _ = asyncio.run(session.send("arrange this as a trio, together"))
    validate_abc(update.abc)
    voices = voice_bars(update.abc)
    assert len(voices) == 3
    assert len({len(bars) for bars in voices.values()}) == 1
    assert update.abc.count("%%MIDI program") == 3
