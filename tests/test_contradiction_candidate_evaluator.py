import pytest

from src.evaluation.contradiction_candidate_evaluator import evaluate_candidate_recall


def pair(left, right, pair_id):
    return {"pair_id": pair_id, "paper1_id": left, "paper2_id": right}


def candidate(left, right):
    return {"paper1": {"id": left}, "paper2": {"id": right}}


def test_candidate_metrics_recover_reversed_pair_and_report_missing():
    references = [pair("A", "B", "C1"), pair("C", "D", "C2")]
    candidates = [candidate("B", "A")]

    result = evaluate_candidate_recall(references, candidates, k_values=[1, 5])

    assert result["recall@1"] == 0.5
    assert result["recall@5"] == 0.5
    assert result["full_pool_recall"] == 0.5
    assert result["recovered_count"] == 1
    assert result["missing_reference_ids"] == ["C2"]


def test_candidate_evaluator_rejects_duplicate_unordered_candidates():
    with pytest.raises(ValueError, match="Duplicate unordered pair"):
        evaluate_candidate_recall(
            [pair("A", "B", "C1")],
            [candidate("A", "B"), candidate("B", "A")],
            k_values=[1],
        )


def test_candidate_evaluator_rejects_invalid_cutoff():
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_candidate_recall([pair("A", "B", "C1")], [], k_values=[0])
