from llmcomposer.models import ScoreUpdate

RAW = "X:1\nT:t\nM:4/4\nL:1/8\nK:C\nCDEF |]\n"


def test_score_update_passthrough():
    update = ScoreUpdate(reply="hi", abc=RAW)
    assert update.abc == RAW.strip()


def test_score_update_strips_markdown_fences():
    fenced = f"```abc\n{RAW}```"
    update = ScoreUpdate(reply="hi", abc=fenced)
    assert "```" not in update.abc
    assert update.abc.startswith("X:1")
