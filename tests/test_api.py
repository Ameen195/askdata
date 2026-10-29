from app import api


def test_health():
    assert api.health() == {"status": "ok"}


def test_ask_with_mocked_graph(monkeypatch):
    monkeypatch.setattr(api.graph, "invoke", lambda state: {
        "answer": "There are 3.", "intent": "SQL", "sql": "SELECT 3", "rows": [{"n": 3}],
        "chunks": [], "grounded": True, "trace": [{"node": "router", "ms": 1}],
    })
    monkeypatch.setattr(api, "save_trace", lambda *args: None)
    response = api.ask(api.Ask(question="count orders"))
    assert response["answer"] == "There are 3."
    assert response["grounded"] is True
