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
