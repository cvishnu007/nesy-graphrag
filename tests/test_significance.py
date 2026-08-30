import csv

import pytest

from src.evaluation.significance import load_paired_deltas, paired_significance


def test_paired_significance_detects_consistent_improvement():
    result = paired_significance([0.1, 0.2, 0.1, 0.2], bootstrap_samples=1000, seed=1)
    assert result["mean_delta"] == pytest.approx(0.15)
    assert result["bootstrap_95_ci"][0] > 0
    assert result["bootstrap_probability_positive"] == 1.0


def test_paired_significance_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one"):
        paired_significance([])


def test_load_paired_deltas(tmp_path):
    path = tmp_path / "metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["query_id", "method", "ndcg@10"])
        writer.writeheader()
        writer.writerows(
            [
                {"query_id": "Q1", "method": "vector", "ndcg@10": 0.5},
                {"query_id": "Q1", "method": "hybrid", "ndcg@10": 0.7},
                {"query_id": "Q2", "method": "vector", "ndcg@10": 0.4},
                {"query_id": "Q2", "method": "hybrid", "ndcg@10": 0.3},
            ]
        )
    assert load_paired_deltas(path, "hybrid", "vector", "ndcg@10") == pytest.approx([0.2, -0.1])
