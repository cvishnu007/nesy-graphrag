from src.pipeline import orchestrator


def test_get_llm_constructs_selected_client_once(monkeypatch):
    fake_client = object()
    calls = []

    def fake_factory():
        calls.append("created")
        return fake_client

    monkeypatch.setattr(orchestrator, "_llm_client", None, raising=False)
    monkeypatch.setattr(orchestrator, "create_llm_client", fake_factory, raising=False)

    assert orchestrator.get_llm() is fake_client
    assert orchestrator.get_llm() is fake_client
    assert calls == ["created"]


def test_graphrag_query_passes_generic_client_to_review(monkeypatch):
    fake_client = object()
    fake_driver = object()
    received = {}
    monkeypatch.setattr(orchestrator, "get_llm", lambda: fake_client, raising=False)
    monkeypatch.setattr(orchestrator, "get_neo4j", lambda: fake_driver)

    def fake_review(client, driver, query, *, top_k):
        received.update(
            client=client,
            driver=driver,
            query=query,
            top_k=top_k,
        )
        return {"papers": []}

    monkeypatch.setattr(orchestrator, "llm_review", fake_review)

    result = orchestrator.graphrag_query("question", mode="review", top_k=4)

    assert result == {"papers": []}
    assert received == {
        "client": fake_client,
        "driver": fake_driver,
        "query": "question",
        "top_k": 4,
    }
