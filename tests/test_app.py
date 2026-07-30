import json

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from llmcomposer.abc_notation import validate_abc
from llmcomposer.app import SESSION_COOKIE, SessionStore, create_app

BORROWED_ABC = "X:1\nT:borrowed\nM:4/4\nL:1/8\nQ:1/4=80\nK:Em\nEF GA B2 cd | e8 |]\n"


def make_client() -> TestClient:
    return TestClient(create_app(model="offline"))


def stream_events(client: TestClient, message: str) -> list[dict]:
    with client.stream("POST", "/chat/stream", json={"message": message}) as res:
        assert res.status_code == 200
        return [
            json.loads(line[len("data: ") :])
            for line in res.iter_lines()
            if line.startswith("data: ")
        ]


def test_index_serves_page_and_seats_the_visitor():
    with make_client() as client:
        res = client.get("/")
    assert res.status_code == 200
    assert "llm<em>composer</em>" in res.text
    assert res.cookies.get(SESSION_COOKIE)


def test_static_assets_are_served():
    with make_client() as client:
        res = client.get("/static/abcjs-basic-min.js")
    assert res.status_code == 200
    assert len(res.content) > 1000


def test_chat_returns_reply_and_valid_score():
    with make_client() as client:
        res = client.post("/chat", json={"message": "like rain on a window"})
        assert res.status_code == 200
        data = res.json()
        assert data["reply"]
        validate_abc(data["abc"])
        assert data["meta"]["bounces"] == []
        assert data["meta"]["prompt_version"]

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
    with make_client() as client:
        events = stream_events(client, "a bright morning")
    assert events[0]["type"] == "writing"
    assert events[-1]["type"] == "final"
    validate_abc(events[-1]["abc"])


def key_of(abc: str) -> str:
    return next(line for line in abc.splitlines() if line.startswith("K:"))


def test_two_visitors_compose_independently():
    app = create_app(model="offline")
    with TestClient(app) as one, TestClient(app) as two:
        one.get("/")
        two.get("/")
        assert one.cookies.get(SESSION_COOKIE) != two.cookies.get(SESSION_COOKIE)

        rain = one.post("/chat", json={"message": "like rain on a window"}).json()
        dawn = two.post("/chat", json={"message": "a bright morning"}).json()
        assert key_of(rain["abc"]) != key_of(dawn["abc"])
        # Each follow-up must revise its own score, not the other visitor's.
        rainy = one.post("/chat", json={"message": "keep going"}).json()["abc"]
        bright = two.post("/chat", json={"message": "keep going"}).json()["abc"]
    assert key_of(rainy) == key_of(rain["abc"])
    assert key_of(bright) == key_of(dawn["abc"])


def test_reset_only_clears_the_caller_s_session():
    app = create_app(model="offline")
    with TestClient(app) as one, TestClient(app) as two:
        rain = one.post("/chat", json={"message": "like rain on a window"}).json()
        two.post("/chat", json={"message": "a bright morning"})
        assert two.post("/reset").json() == {"ok": True}
        rainy = one.post("/chat", json={"message": "keep going"}).json()["abc"]
    assert key_of(rainy) == key_of(rain["abc"])


def test_score_endpoint_makes_a_tune_the_working_score():
    with make_client() as client:
        client.post("/chat", json={"message": "like rain on a window"})
        res = client.post("/score", json={"abc": BORROWED_ABC})
        assert res.status_code == 200
        assert res.json() == {"ok": True}
        # The next turn revises the score we handed back, not the old one.
        revised = client.post("/chat", json={"message": "keep going"}).json()
    validate_abc(revised["abc"])
    assert "K:Em" in revised["abc"]


def test_score_endpoint_refuses_invalid_abc_with_a_readable_reason():
    with make_client() as client:
        res = client.post("/score", json={"abc": "not a score at all"})
    assert res.status_code == 422
    detail = res.json()["detail"]
    assert isinstance(detail, str)
    assert detail


def test_stream_reports_a_mid_flight_failure_as_a_final_frame():
    async def stream(messages: list[ModelMessage], info: AgentInfo):
        raise RuntimeError("the provider went quiet")
        yield {}  # pragma: no cover - unreachable, makes this a generator

    model = FunctionModel(stream_function=stream, model_name="boom")
    with TestClient(create_app(model=model)) as client:
        events = stream_events(client, "anything")
    assert events[-1]["type"] == "error"
    assert "quiet" in events[-1]["detail"]


def test_session_store_evicts_the_least_recently_used_seat():
    store = SessionStore(model="offline", capacity=2)
    first, _, _ = store.seat(None)
    second, _, _ = store.seat(None)
    store.seat(first)  # touch, so the second is now the stale one
    third, _, _ = store.seat(None)

    assert len(store) == 2
    assert store.seat(first)[0] == first
    assert store.seat(third)[0] == third
    assert store.seat(second)[0] != second  # evicted, so a fresh seat is minted


def test_an_unknown_cookie_never_seats_itself():
    store = SessionStore(model="offline")
    minted, _, _ = store.seat("i-picked-this-myself")
    assert minted != "i-picked-this-myself"
