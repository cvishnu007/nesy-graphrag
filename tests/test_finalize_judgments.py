import csv

import pytest

from src.evaluation.benchmark_io import BenchmarkValidationError
from src.evaluation.finalize_judgments import (
    load_judgments,
    merge_judgments,
)


FIELDS = [
    "query_id",
    "split",
    "query",
    "paper_id",
    "grade",
    "notes",
]


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(query_id="Q001", paper_id="p1", grade="2"):
    return {
        "query_id": query_id,
        "split": "dev",
        "query": "query text",
        "paper_id": paper_id,
        "grade": grade,
        "notes": "Directly relevant.",
    }


def _benchmark():
    return {
        "benchmark_version": "0.1-draft",
        "status": "pending",
        "corpus": {
            "source": "s2",
            "paper_count": 10,
            "years": "2020-2025",
            "collection": "papers",
        },
        "queries": [
            {
                "query_id": f"Q{index:03d}",
                "split": "dev" if index <= 6 else "test",
                "query": f"query {index}",
                "judgments": {},
            }
            for index in range(1, 21)
        ],
    }


def test_load_judgments_accepts_complete_rows(tmp_path):
    path = tmp_path / "judgments.csv"
    _write_csv(path, [_row()])
    assert load_judgments(path) == {"Q001": {"p1": 2}}


@pytest.mark.parametrize("grade", ["", "3", "1.0", "relevant"])
def test_load_judgments_rejects_invalid_grade(tmp_path, grade):
    path = tmp_path / "judgments.csv"
    _write_csv(path, [_row(grade=grade)])
    with pytest.raises(BenchmarkValidationError, match="invalid grade"):
        load_judgments(path)


def test_load_judgments_rejects_duplicate_pair(tmp_path):
    path = tmp_path / "judgments.csv"
    _write_csv(path, [_row(), _row()])
    with pytest.raises(BenchmarkValidationError, match="Duplicate judgment"):
        load_judgments(path)


def test_merge_judgments_marks_review_pending_draft():
    benchmark = _benchmark()
    judgments = {
        f"Q{index:03d}": {f"p{index}": 2}
        for index in range(1, 21)
    }
    merged = merge_judgments(benchmark, judgments)
    assert merged["benchmark_version"] == "0.2-draft"
    assert merged["status"] == "judgments_pending_human_review"
    assert merged["judgment_metadata"]["human_review_required"] is True
    assert merged["queries"][0]["judgments"] == {"p1": 2}
    assert benchmark["queries"][0]["judgments"] == {}


def test_merge_judgments_rejects_missing_query():
    benchmark = _benchmark()
    judgments = {
        f"Q{index:03d}": {f"p{index}": 2}
        for index in range(1, 20)
    }
    with pytest.raises(BenchmarkValidationError, match="missing=.*Q020"):
        merge_judgments(benchmark, judgments)
