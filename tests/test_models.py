from llmcomposer.models import Bounce, ScoreRequest, ScoreUpdate, TurnMeta

RAW = "X:1\nT:t\nM:4/4\nL:1/8\nK:C\nCDEF |]\n"


def test_score_update_passthrough():
    update = ScoreUpdate(reply="hi", abc=RAW)
    assert update.abc == RAW.strip()


def test_score_update_strips_markdown_fences():
    fenced = f"```abc\n{RAW}```"
    update = ScoreUpdate(reply="hi", abc=fenced)
    assert "```" not in update.abc
    assert update.abc.startswith("X:1")


def test_score_request_carries_raw_abc():
    assert ScoreRequest(abc=RAW).abc == RAW


def test_turn_meta_defaults_are_empty_not_missing():
    meta = TurnMeta(model="offline")
    assert meta.prompt_version == ""
    assert meta.prompt_sha == ""
    assert meta.bounces == []
    assert meta.corrections == 0
    assert meta.temperature is None
    assert meta.seed is None


def test_turn_meta_carries_bounces_and_sampling_settings():
    bounce = Bounce(
        attempt=1,
        code="bar_length",
        message="voice 1 bar 3 is short",
        rejected_abc=RAW,
    )
    meta = TurnMeta(
        model="offline",
        prompt_version="composer-v3",
        prompt_sha="0123456789abcdef",
        corrections=1,
        bounces=[bounce],
        temperature=0.7,
        seed=11,
    )
    assert meta.corrections == len(meta.bounces)
    dumped = meta.model_dump()
    assert dumped["bounces"][0]["code"] == "bar_length"
    assert dumped["bounces"][0]["rejected_abc"] == RAW
    assert dumped["prompt_version"] == "composer-v3"
    assert dumped["prompt_sha"] == "0123456789abcdef"
    assert dumped["temperature"] == 0.7
    assert dumped["seed"] == 11
