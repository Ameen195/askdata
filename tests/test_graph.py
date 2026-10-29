from app import graph as graph_module


def test_router_json_parsing_is_used(monkeypatch):
    monkeypatch.setattr(graph_module, "run_prompt", lambda *args, **kwargs: {"intent": "SQL"})
    monkeypatch.setattr(graph_module, "get_schema", lambda: "orders(order_id INT)")
    result = graph_module.router({"question": "count orders"})
    assert result["intent"] == "SQL"
