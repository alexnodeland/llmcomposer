from fastapi.testclient import TestClient

from llmcomposer.abc_notation import validate_abc
from llmcomposer.app import create_app


def make_client() -> TestClient:
    return TestClient(create_app(model="offline"))


def test_index_serves_page():
    with make_client() as client:
        res = client.get("/")
    assert res.status_code == 200
    assert "llm<em>composer</em>" in res.text


def test_chat_returns_reply_and_valid_score():
    with make_client() as client:
        res = client.post("/chat", json={"message": "like rain on a window"})
        assert res.status_code == 200
        data = res.json()
        assert data["reply"]
        validate_abc(data["abc"])

        reset = client.post("/reset")
        assert reset.json() == {"ok": True}


def test_chat_rejects_empty_message():
    with make_client() as client:
        res = client.post("/chat", json={"message": ""})
    assert res.status_code == 422


def test_models_endpoint_lists_and_switches(monkeypatch):
    monkeypatch.setenv("LLMCOMPOSER_MODELS", "offline")
    with make_client() as client:
        listing = client.get("/models").json()
        assert listing["models"] == ["offline"]

        res = client.post("/model", json={"model": "offline"})
        assert res.status_code == 200
        assert "offline-composer" in res.json()["model"]

        unknown = client.post("/model", json={"model": "nope:nothere"})
        assert unknown.status_code == 404


def test_model_switch_keeps_the_score(monkeypatch):
    monkeypatch.setenv("LLMCOMPOSER_MODELS", "offline")
    with make_client() as client:
        first = client.post("/chat", json={"message": "like rain on a window"})
        client.post("/model", json={"model": "offline"})
        revised = client.post("/chat", json={"message": "make it slower"})
    assert first.status_code == 200
    assert revised.status_code == 200
    validate_abc(revised.json()["abc"])


def test_chat_stream_emits_events_and_final():
    import json

    with (
        make_client() as client,
        client.stream(
            "POST", "/chat/stream", json={"message": "a bright morning"}
        ) as res,
    ):
        assert res.status_code == 200
        events = [
            json.loads(line[len("data: ") :])
            for line in res.iter_lines()
            if line.startswith("data: ")
        ]
    assert events[0]["type"] == "writing"
    assert events[-1]["type"] == "final"
    validate_abc(events[-1]["abc"])
