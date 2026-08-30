import csv
import json

import pytest

from src.evaluation.candidate_pool import (
    pool_benchmark,
    pool_query,
    write_pool_outputs,
)


def paper(paper_id):
    return {
        "id": paper_id,
        "title": f"Title {paper_id}",
        "abstract": f"Abstract {paper_id}",
        "year": 2024,
        "category": "Computer Science",
        "score": 1.0,
        "source": "test",
    }


def make_retrievers():
    method_rows = {
        "bm25": [paper("p1"), paper("p2")],
        "vector": [paper("p2"), paper("p3")],
        "graph": [paper("p4")],
        "hybrid": [paper("p1"), paper("p4")],
    }

    return {
        method: (
            lambda query, top_k, rows=rows: [
                dict(row) for row in rows[:top_k]
            ]
        )
        for method, rows in method_rows.items()
    }


def query_record(query_id="Q001", split="dev"):
    return {
        "query_id": query_id,
        "split": split,
        "query": "graph neural networks",
        "judgments": {},
    }


def test_pool_query_deduplicates_and_method_hides_candidates():
    method_hidden, audit, summary = pool_query(
        query_record(),
        make_retrievers(),
        benchmark_version="0.1-draft",
        pool_depth=20,
    )

    assert len(method_hidden) == 4
    assert len({row["paper_id"] for row in method_hidden}) == 4
    assert all("methods" not in row for row in method_hidden)
    assert all("ranks" not in row for row in method_hidden)
    assert all(row["grade"] == "" for row in method_hidden)
    assert len(audit) == 4
    assert summary["unique_candidates"] == 4
    assert summary["method_counts"] == {
        "bm25": 2,
        "vector": 2,
        "graph": 1,
        "hybrid": 2,
    }


def test_method_hidden_order_is_deterministic():
    first, _, _ = pool_query(
        query_record(),
        make_retrievers(),
        benchmark_version="0.1-draft",
        pool_depth=20,
    )
    second, _, _ = pool_query(
        query_record(),
        make_retrievers(),
        benchmark_version="0.1-draft",
        pool_depth=20,
    )

    assert first == second


def test_missing_result_field_is_rejected():
    retrievers = make_retrievers()
    invalid = paper("bad")
    del invalid["abstract"]
    retrievers["bm25"] = lambda query, top_k: [invalid]

    with pytest.raises(RuntimeError):
        pool_query(
            query_record(),
            retrievers,
            benchmark_version="0.1-draft",
            pool_depth=20,
        )


def test_duplicate_id_from_one_method_is_rejected():
    retrievers = make_retrievers()
    retrievers["graph"] = lambda query, top_k: [
        paper("duplicate"),
        paper("duplicate"),
    ]

    with pytest.raises(RuntimeError):
        pool_query(
            query_record(),
            retrievers,
            benchmark_version="0.1-draft",
            pool_depth=20,
        )


@pytest.mark.parametrize("invalid_depth", [19, 31])
def test_invalid_pool_depth_is_rejected(invalid_depth):
    with pytest.raises(ValueError):
        pool_query(
            query_record(),
            make_retrievers(),
            benchmark_version="0.1-draft",
            pool_depth=invalid_depth,
        )


def test_retriever_failure_is_not_hidden():
    retrievers = make_retrievers()

    def fail(query, top_k):
        raise ConnectionError("service unavailable")

    retrievers["vector"] = fail

    with pytest.raises(RuntimeError, match="Q001.*vector"):
        pool_query(
            query_record(),
            retrievers,
            benchmark_version="0.1-draft",
            pool_depth=20,
        )


def test_pool_benchmark_combines_queries():
    benchmark = {
        "benchmark_version": "0.1-draft",
        "queries": [
            query_record("Q001", "dev"),
            query_record("Q002", "test"),
        ],
    }

    method_hidden, audit, summary = pool_benchmark(
        benchmark,
        make_retrievers(),
        pool_depth=20,
    )

    assert len(method_hidden) == 8
    assert len(audit) == 8
    assert summary["query_count"] == 2
    assert summary["candidate_count"] == 8


def test_write_outputs_and_prevent_overwrite(tmp_path):
    method_hidden, audit, summary = pool_query(
        query_record(),
        make_retrievers(),
        benchmark_version="0.1-draft",
        pool_depth=20,
    )
    full_summary = {
        "benchmark_version": "0.1-draft",
        "query_count": 1,
        "candidate_count": len(method_hidden),
    }
    judgments_path = tmp_path / "judgments.csv"
    audit_path = tmp_path / "audit.jsonl"
    summary_path = tmp_path / "summary.json"

    write_pool_outputs(
        method_hidden,
        audit,
        full_summary,
        judgments_path=judgments_path,
        audit_path=audit_path,
        summary_path=summary_path,
    )

    with judgments_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert "methods" not in rows[0]
    assert rows[0]["grade"] == ""

    audit_rows = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(audit_rows) == 4
    assert "methods" in audit_rows[0]
    assert json.loads(summary_path.read_text(encoding="utf-8")) == full_summary

    with pytest.raises(FileExistsError):
        write_pool_outputs(
            method_hidden,
            audit,
            full_summary,
            judgments_path=judgments_path,
            audit_path=audit_path,
            summary_path=summary_path,
        )
