import pytest

from src.evaluation.ir_metrics import (
    aggregate_query_metrics,
    average_precision,
    evaluate_ranking,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    unjudged_rate_at_k,
)


def test_perfect_ranking():
    ranked = ["p1", "p2", "p3"]
    judgments = {"p1": 2, "p2": 1, "p3": 0}

    assert precision_at_k(ranked, judgments, 2) == 1.0
    assert recall_at_k(ranked, judgments, 2) == 1.0
    assert hit_rate_at_k(ranked, judgments, 2) == 1.0
    assert reciprocal_rank(ranked, judgments) == 1.0
    assert average_precision(ranked, judgments) == 1.0
    assert ndcg_at_k(ranked, judgments, 2) == pytest.approx(1.0)


def test_first_relevant_paper_at_rank_three():
    ranked = ["wrong-1", "wrong-2", "relevant"]
    judgments = {"relevant": 1}

    assert reciprocal_rank(ranked, judgments) == pytest.approx(1 / 3)
    assert hit_rate_at_k(ranked, judgments, 2) == 0.0
    assert hit_rate_at_k(ranked, judgments, 3) == 1.0


def test_no_relevant_result():
    ranked = ["p1", "p2"]
    judgments = {"p3": 2}

    assert precision_at_k(ranked, judgments, 2) == 0.0
    assert recall_at_k(ranked, judgments, 2) == 0.0
    assert reciprocal_rank(ranked, judgments) == 0.0
    assert average_precision(ranked, judgments) == 0.0


def test_empty_ranking_returns_zero():
    judgments = {"p1": 2}

    assert precision_at_k([], judgments, 5) == 0.0
    assert recall_at_k([], judgments, 5) == 0.0
    assert hit_rate_at_k([], judgments, 5) == 0.0
    assert reciprocal_rank([], judgments) == 0.0
    assert average_precision([], judgments) == 0.0
    assert ndcg_at_k([], judgments, 5) == 0.0


def test_no_known_relevant_papers_returns_zero():
    ranked = ["p1"]
    judgments = {"p1": 0}

    assert recall_at_k(ranked, judgments, 1) == 0.0
    assert average_precision(ranked, judgments) == 0.0
    assert ndcg_at_k(ranked, judgments, 1) == 0.0


def test_duplicate_ranked_ids_are_ignored():
    ranked = ["p1", "p1", "p2"]
    judgments = {"p1": 2, "p2": 1}

    assert precision_at_k(ranked, judgments, 2) == 1.0
    assert recall_at_k(ranked, judgments, 2) == 1.0
    assert average_precision(ranked, judgments) == 1.0


def test_graded_ndcg_rewards_best_order():
    judgments = {"high": 2, "partial": 1}

    best = ndcg_at_k(["high", "partial"], judgments, 2)
    reversed_order = ndcg_at_k(["partial", "high"], judgments, 2)

    assert best == pytest.approx(1.0)
    assert reversed_order < best


def test_k_larger_than_ranking():
    ranked = ["p1"]
    judgments = {"p1": 2}

    assert precision_at_k(ranked, judgments, 5) == pytest.approx(0.2)
    assert recall_at_k(ranked, judgments, 5) == 1.0


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_invalid_k_is_rejected(invalid_k):
    with pytest.raises(ValueError):
        precision_at_k(["p1"], {"p1": 2}, invalid_k)


def test_unjudged_rate():
    ranked = ["judged-relevant", "unknown", "judged-not-relevant"]
    judgments = {
        "judged-relevant": 2,
        "judged-not-relevant": 0,
    }

    assert unjudged_rate_at_k(ranked, judgments, 3) == pytest.approx(1 / 3)


def test_evaluate_ranking_returns_all_metrics():
    result = evaluate_ranking(
        ["p1", "p2"],
        {"p1": 2, "p2": 0},
        k_values=(2,),
    )

    assert set(result) == {
        "mrr",
        "map",
        "precision@2",
        "recall@2",
        "hit_rate@2",
        "ndcg@2",
        "unjudged_rate@2",
    }


def test_aggregate_query_metrics():
    result = aggregate_query_metrics(
        [
            {"mrr": 1.0, "map": 0.5},
            {"mrr": 0.0, "map": 1.0},
        ]
    )

    assert result["query_count"] == 2
    assert result["metrics"]["mrr"]["mean"] == pytest.approx(0.5)
    assert result["metrics"]["mrr"]["std"] == pytest.approx(0.5)
    assert result["metrics"]["map"]["mean"] == pytest.approx(0.75)
    assert result["metrics"]["map"]["std"] == pytest.approx(0.25)