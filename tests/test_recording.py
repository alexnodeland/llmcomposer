import json

from llmcomposer.models import Bounce, TurnMeta
from llmcomposer.recording import RECORD_DIR_ENV, Recorder


def make_meta() -> TurnMeta:
    return TurnMeta(
        model="function:offline-composer",
        prompt_version="composer-v3",
        prompt_sha="0123456789abcdef",
        requests=2,
        input_tokens=11,
        output_tokens=22,
        corrections=1,
        bounces=[
            Bounce(
                attempt=1,
                code="bar_length",
                message="bar 3 is short",
                rejected_abc="X:1\nT:t\n",
            )
        ],
        temperature=0.2,
        seed=7,
        elapsed_ms=345,
    )


def read_lines(directory) -> list[dict]:
    files = sorted(directory.glob("turns-*.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text().splitlines()]


def test_recorder_is_a_noop_without_the_env_var(monkeypatch, tmp_path):
    monkeypatch.delenv(RECORD_DIR_ENV, raising=False)
    recorder = Recorder()
    assert recorder.enabled is False
    written = recorder.record(
        session_id="s",
        turn_index=0,
        user_message="hi",
        reply="ok",
        abc="X:1\n",
        meta=make_meta(),
    )
    assert written is None
    assert list(tmp_path.iterdir()) == []


def test_recorder_appends_one_json_line_per_turn(monkeypatch, tmp_path):
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path / "runs"))
    recorder = Recorder()
    assert recorder.enabled is True
    for index in range(2):
        recorder.record(
            session_id="abc123",
            turn_index=index,
            user_message=f"turn {index}",
            reply="ok",
            abc="X:1\nT:t\n",
            meta=make_meta(),
        )

    records = read_lines(tmp_path / "runs")
    assert [record["turn_index"] for record in records] == [0, 1]
    first = records[0]
    assert first["session_id"] == "abc123"
    assert first["model"] == "function:offline-composer"
    assert first["prompt_version"] == "composer-v3"
    assert first["prompt_sha"] == "0123456789abcdef"
    assert first["user_message"] == "turn 0"
    assert first["reply"] == "ok"
    assert first["abc"].startswith("X:1")
    assert first["usage"] == {"requests": 2, "input_tokens": 11, "output_tokens": 22}
    assert first["elapsed_ms"] == 345
    assert first["temperature"] == 0.2
    assert first["seed"] == 7
    assert first["ts"].startswith("20")
    assert first["bounces"] == [
        {
            "attempt": 1,
            "code": "bar_length",
            "message": "bar 3 is short",
            "rejected_abc": "X:1\nT:t\n",
        }
    ]


def test_explicit_directory_beats_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path / "ignored"))
    chosen = tmp_path / "chosen"
    written = Recorder(chosen).record(
        session_id="s",
        turn_index=0,
        user_message="hi",
        reply="ok",
        abc="X:1\n",
        meta=make_meta(),
    )
    assert written is not None
    assert written.parent == chosen
    assert not (tmp_path / "ignored").exists()


def test_a_failed_write_never_breaks_the_turn(monkeypatch, tmp_path):
    blocked = tmp_path / "afile"
    blocked.write_text("not a directory")
    monkeypatch.setenv(RECORD_DIR_ENV, str(blocked))
    written = Recorder().record(
        session_id="s",
        turn_index=0,
        user_message="hi",
        reply="ok",
        abc="X:1\n",
        meta=make_meta(),
    )
    assert written is None
