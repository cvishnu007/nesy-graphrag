from src.storage import chroma_store


class FakeCollection:
    def get(self, ids, include):
        assert include == ["embeddings"]
        available = {
            "same": [1.0, 0.0],
            "opposite": [-1.0, 0.0],
            "zero": [0.0, 0.0],
        }
        found = [paper_id for paper_id in ids if paper_id in available]
        return {
            "ids": found,
            "embeddings": [available[paper_id] for paper_id in found],
        }


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        assert texts == ["query"]
        self.calls += 1
        return [[1.0, 0.0]]


def test_score_papers_reuses_stored_embeddings(monkeypatch):
    chroma_store.encode_query.cache_clear()
    monkeypatch.setattr(chroma_store, "get_collection", lambda: FakeCollection())
    monkeypatch.setattr(chroma_store, "get_embedder", lambda: FakeEmbedder())

    scores = chroma_store.score_papers_against_query(
        "query",
        ["same", "same", "opposite", "zero", "missing"],
    )

    assert scores == {"same": 1.0, "opposite": 0.0}


def test_query_embedding_is_reused_between_retrieval_stages(monkeypatch):
    embedder = FakeEmbedder()
    chroma_store.encode_query.cache_clear()
    monkeypatch.setattr(chroma_store, "get_embedder", lambda: embedder)

    assert chroma_store.encode_query("query") == (1.0, 0.0)
    assert chroma_store.encode_query("query") == (1.0, 0.0)
    assert embedder.calls == 1


def test_score_papers_handles_empty_input_without_loading_models(monkeypatch):
    monkeypatch.setattr(
        chroma_store,
        "get_collection",
        lambda: (_ for _ in ()).throw(AssertionError("collection should not load")),
    )

    assert chroma_store.score_papers_against_query("query", []) == {}
    assert chroma_store.score_papers_against_query("   ", ["paper"]) == {}
