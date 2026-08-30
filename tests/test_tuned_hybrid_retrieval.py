import pytest

from src.pipeline.tuned_hybrid_retrieval import weighted_rrf_fuse


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
        "bm25": [paper("shared"), paper("lexical")],
        "vector": [paper("semantic"), paper("shared")],
        "graph": [paper("graph"), paper("shared")],
    }


def test_overlap_is_rewarded_and_results_are_unique():
    results = weighted_rrf_fuse(
        rankings(),
        {"bm25": 2.0, "vector": 1.0, "graph": 1.0},
        top_k=4,
        rrf_k=60,
    )
    assert results[0]["id"] == "shared"
    assert len({row["id"] for row in results}) == len(results)
    assert results[0]["component_ranks"] == {
        "bm25": 1,
        "vector": 2,
        "graph": 2,
    }


def test_bm25_receives_the_frozen_double_weight():
    results = weighted_rrf_fuse(
        {
            "bm25": [paper("lexical")],
            "vector": [paper("semantic")],
            "graph": [paper("graph")],
        },
        {"bm25": 2.0, "vector": 1.0, "graph": 1.0},
        top_k=3,
        rrf_k=60,
    )
    assert results[0]["id"] == "lexical"


def test_fusion_is_deterministic():
    first = weighted_rrf_fuse(
        rankings(), {"bm25": 2, "vector": 1, "graph": 1}, top_k=4
    )
    second = weighted_rrf_fuse(
        rankings(), {"bm25": 2, "vector": 1, "graph": 1}, top_k=4
    )
    assert first == second


def test_invalid_weights_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        weighted_rrf_fuse(
            rankings(), {"bm25": 0, "vector": 0, "graph": 0}, top_k=4
        )


def test_duplicate_component_ids_are_rejected():
    duplicate = rankings()
    duplicate["bm25"] = [paper("same"), paper("same")]
    with pytest.raises(ValueError, match="duplicate"):
        weighted_rrf_fuse(
            duplicate, {"bm25": 2, "vector": 1, "graph": 1}, top_k=4
        )
