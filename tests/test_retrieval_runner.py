import json

import pytest

from src.evaluation.retrieval_runner import evaluate_benchmark, write_outputs


def _benchmark():
    return {
        "benchmark_version": "test",
        "status": "judged",
        "primary_metric": "ndcg@10",
        "secondary_metric": "recall@10",
        "queries": [
            {
                "query_id": "Q001",
                "split": "dev",
                "query": "graph query",
                "judgments": {"relevant": 2, "partial": 1, "bad": 0},
            }
        ],
    }


def _retrievers():
    good = [{"id": "relevant"}, {"id": "partial"}]
    bad = [{"id": "bad"}, {"id": "unknown"}]
    return {
        "bm25": lambda query, top_k: bad,
        "vector": lambda query, top_k: good,
        "graph": lambda query, top_k: bad,
        "hybrid": lambda query, top_k: good,
    }


def test_evaluate_benchmark_compares_all_methods():
    rows, rankings, summary = evaluate_benchmark(
        _benchmark(), _retrievers(), top_k=20
    )
    assert len(rows) == 4
    assert len(rankings) == 4
    assert {row["method"] for row in rows} == {
        "bm25", "vector", "graph", "hybrid"
    }
    assert summary["query_count"] == 1
    assert summary["winner_by_ndcg_at_10"] in {"vector", "hybrid"}
    vector = next(row for row in rows if row["method"] == "vector")
    bm25 = next(row for row in rows if row["method"] == "bm25")
    assert vector["ndcg@10"] > bm25["ndcg@10"]


def test_evaluate_benchmark_rejects_duplicate_results():
    retrievers = _retrievers()
    retrievers["bm25"] = lambda query, top_k: [{"id": "same"}, {"id": "same"}]
    with pytest.raises(RuntimeError, match="duplicate paper ID"):
        evaluate_benchmark(_benchmark(), retrievers, top_k=20)


def test_evaluate_benchmark_rejects_small_top_k():
    with pytest.raises(ValueError, match="at least 20"):
        evaluate_benchmark(_benchmark(), _retrievers(), top_k=10)


def test_write_outputs_creates_three_artifacts(tmp_path):
    rows, rankings, summary = evaluate_benchmark(
        _benchmark(), _retrievers(), top_k=20
    )
    write_outputs(rows, rankings, summary, tmp_path)
    assert (tmp_path / "per_query_metrics.csv").exists()
    assert len((tmp_path / "rankings.jsonl").read_text().splitlines()) == 4
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["winner_by_ndcg_at_10"] == summary["winner_by_ndcg_at_10"]


def test_write_outputs_refuses_overwrite(tmp_path):
    rows, rankings, summary = evaluate_benchmark(
        _benchmark(), _retrievers(), top_k=20
    )
    write_outputs(rows, rankings, summary, tmp_path)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_outputs(rows, rankings, summary, tmp_path)
