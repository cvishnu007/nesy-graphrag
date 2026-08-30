import pytest

from src.evaluation.retrievers import two_way_hybrid_retrieval as hybrid_module
from src.evaluation.retrievers.two_way_hybrid_retrieval import (
    evaluation_hybrid_retrieve,
    weighted_rrf_fuse,
)


def paper(paper_id):
    return {
        "id": paper_id,
        "title": paper_id,
        "abstract": "abstract",
        "year": 2024,
        "category": "Computer Science",
        "score": 1.0,
        "source": "test",
    }


def rankings():
    return {
        "vector": [paper("semantic"), paper("shared")],
        "graph": [paper("graph"), paper("shared")],
    }


def test_overlap_is_rewarded_and_results_are_unique():
    results = weighted_rrf_fuse(
        rankings(),
        {"vector": 1.0, "graph": 1.0},
        top_k=4,
        rrf_k=60,
    )
    assert results[0]["id"] == "shared"
    assert len({row["id"] for row in results}) == len(results)
    assert results[0]["component_ranks"] == {
        "vector": 2,
        "graph": 2,
    }


def test_vector_receives_the_frozen_higher_weight():
    results = weighted_rrf_fuse(
        {
            "vector": [paper("semantic")],
            "graph": [paper("graph")],
        },
        {"vector": 16.0, "graph": 1.0},
        top_k=2,
        rrf_k=60,
    )
    assert results[0]["id"] == "semantic"


def test_fusion_is_deterministic():
    first = weighted_rrf_fuse(
        rankings(), {"vector": 16, "graph": 1}, top_k=4
    )
    second = weighted_rrf_fuse(
        rankings(), {"vector": 16, "graph": 1}, top_k=4
    )
    assert first == second


def test_invalid_weights_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        weighted_rrf_fuse(
            rankings(), {"vector": 0, "graph": 0}, top_k=4
        )


def test_duplicate_component_ids_are_rejected():
    duplicate = rankings()
    duplicate["vector"] = [paper("same"), paper("same")]
    with pytest.raises(ValueError, match="duplicate"):
        weighted_rrf_fuse(
            duplicate, {"vector": 16, "graph": 1}, top_k=4
        )


def test_evaluation_hybrid_uses_vector_seeds_for_citation_expansion(monkeypatch):
    vector_rows = [paper("vector-1"), paper("vector-2")]
    graph_rows = [paper("graph-1")]
    calls = {}

    monkeypatch.setattr(
        hybrid_module,
        "vector_only_retrieve",
        lambda query, top_k: list(vector_rows),
    )

    def fake_symbolic_expand(driver, paper_ids):
        calls["driver"] = driver
        calls["paper_ids"] = paper_ids
        return list(graph_rows)

    monkeypatch.setattr(hybrid_module, "symbolic_expand", fake_symbolic_expand)
    driver = object()
    results = evaluation_hybrid_retrieve(driver, "query", top_k=3)

    assert calls == {
        "driver": driver,
        "paper_ids": ["vector-1", "vector-2"],
    }
    assert [row["id"] for row in results] == [
        "vector-1",
        "vector-2",
        "graph-1",
    ]
