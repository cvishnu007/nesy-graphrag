"""Create a method-hidden candidate pool from all retrieval methods."""

import argparse
import csv
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from src.evaluation.benchmark_io import load_benchmark
from src.evaluation.retrievers.bm25_retrieval import bm25_retrieve
from src.evaluation.retrievers.graph_only_retrieval import graph_only_retrieve
from src.pipeline.retrieval import (
    nesy_retrieve,
    vector_only_retrieve,
)
from src.storage.neo4j_store import get_driver


METHOD_NAMES = ("bm25", "vector", "graph", "hybrid")
REQUIRED_RESULT_FIELDS = {
    "id",
    "title",
    "abstract",
    "year",
    "category",
    "score",
    "source",
}


def _validate_pool_depth(pool_depth: int) -> None:
    if not 20 <= pool_depth <= 30:
        raise ValueError(
            "pool_depth must be between 20 and 30"
        )


def _validate_result(
    paper: Mapping[str, Any],
    method: str,
) -> None:
    if not isinstance(paper, Mapping):
        raise RuntimeError(
            f"{method} returned a non-dictionary result"
        )

    missing = sorted(
        REQUIRED_RESULT_FIELDS - set(paper)
    )
    if missing:
        raise RuntimeError(
            f"{method} result is missing fields: "
            f"{', '.join(missing)}"
        )

    if not str(paper["id"]).strip():
        raise RuntimeError(
            f"{method} returned an empty paper ID"
        )


def _method_hidden_sort_key(
    benchmark_version: str,
    query_id: str,
    paper_id: str,
) -> str:
    """Create a stable hidden ordering unrelated to method rank."""
    value = (
        f"{benchmark_version}|{query_id}|{paper_id}"
    )
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def build_retrievers(driver) -> dict[str, Callable]:
    """Create four retrievers with the same two-argument interface."""
    return {
        "bm25": lambda query, top_k: bm25_retrieve(
            query,
            top_k=top_k,
        ),
        "vector": lambda query, top_k: vector_only_retrieve(
            query,
            top_k=top_k,
        ),
        "graph": lambda query, top_k: graph_only_retrieve(
            driver,
            query,
            top_k=top_k,
        ),
        "hybrid": lambda query, top_k: nesy_retrieve(driver, query, top_k=top_k),
    }


def pool_query(
    query_record: Mapping[str, Any],
    retrievers: Mapping[str, Callable],
    *,
    benchmark_version: str,
    pool_depth: int,
) -> tuple[list[dict], list[dict], dict]:
    """Pool results while hiding method identity and original rank."""
    _validate_pool_depth(pool_depth)

    query_id = query_record["query_id"]
    query_text = query_record["query"]
    pooled = {}
    method_counts = {}

    for method in METHOD_NAMES:
        if method not in retrievers:
            raise RuntimeError(
                f"Missing retriever: {method}"
            )

        try:
            results = list(
                retrievers[method](
                    query_text,
                    pool_depth,
                )
            )
        except Exception as error:
            raise RuntimeError(
                f"Pooling failed for {query_id} "
                f"using {method}: {error}"
            ) from error

        method_counts[method] = len(results)
        method_seen_ids = set()

        for rank, paper in enumerate(results, start=1):
            _validate_result(paper, method)
            paper_id = str(paper["id"]).strip()

            if paper_id in method_seen_ids:
                raise RuntimeError(
                    f"{method} returned duplicate paper ID "
                    f"{paper_id} for {query_id}"
                )
            method_seen_ids.add(paper_id)

            if paper_id not in pooled:
                pooled[paper_id] = {
                    "paper_id": paper_id,
                    "title": str(
                        paper.get("title", "") or ""
                    ),
                    "abstract": str(
                        paper.get("abstract", "") or ""
                    ),
                    "year": paper.get("year", ""),
                    "category": str(
                        paper.get("category", "") or ""
                    ),
                    "methods": [],
                    "ranks": {},
                }

            candidate = pooled[paper_id]

            for field in (
                "title",
                "abstract",
                "year",
                "category",
            ):
                if not candidate[field] and paper.get(field):
                    candidate[field] = paper[field]

            candidate["methods"].append(method)
            candidate["ranks"][method] = rank

    method_hidden_candidates = []
    audit_candidates = []

    ordered_candidates = sorted(
        pooled.values(),
        key=lambda candidate: _method_hidden_sort_key(
            benchmark_version,
            query_id,
            candidate["paper_id"],
        ),
    )

    for candidate in ordered_candidates:
        method_hidden_candidates.append(
            {
                "query_id": query_id,
                "split": query_record["split"],
                "query": query_text,
                "paper_id": candidate["paper_id"],
                "title": candidate["title"],
                "abstract": candidate["abstract"],
                "year": candidate["year"],
                "category": candidate["category"],
                "grade": "",
                "notes": "",
            }
        )

        audit_candidates.append(
            {
                "query_id": query_id,
                "paper_id": candidate["paper_id"],
                "methods": sorted(
                    candidate["methods"]
                ),
                "ranks": {
                    method: candidate["ranks"][method]
                    for method in METHOD_NAMES
                    if method in candidate["ranks"]
                },
            }
        )

    summary = {
        "query_id": query_id,
        "split": query_record["split"],
        "method_counts": method_counts,
        "unique_candidates": len(method_hidden_candidates),
    }

    return (
        method_hidden_candidates,
        audit_candidates,
        summary,
    )


def pool_benchmark(
    benchmark: Mapping[str, Any],
    retrievers: Mapping[str, Callable],
    *,
    pool_depth: int = 20,
) -> tuple[list[dict], list[dict], dict]:
    """Pool candidates for every frozen benchmark query."""
    _validate_pool_depth(pool_depth)

    all_method_hidden = []
    all_audit = []
    query_summaries = []
    benchmark_version = benchmark["benchmark_version"]

    total_queries = len(benchmark["queries"])

    for number, query_record in enumerate(
        benchmark["queries"],
        start=1,
    ):
        print(
            f"[{number}/{total_queries}] "
            f"{query_record['query_id']}: "
            f"{query_record['query']}"
        )

        method_hidden, audit, summary = pool_query(
            query_record,
            retrievers,
            benchmark_version=benchmark_version,
            pool_depth=pool_depth,
        )

        all_method_hidden.extend(method_hidden)
        all_audit.extend(audit)
        query_summaries.append(summary)

        print(
            f"  BM25={summary['method_counts']['bm25']} "
            f"Vector={summary['method_counts']['vector']} "
            f"Graph={summary['method_counts']['graph']} "
            f"Hybrid={summary['method_counts']['hybrid']} "
            f"Unique={summary['unique_candidates']}"
        )

    summary = {
        "benchmark_version": benchmark_version,
        "pool_depth": pool_depth,
        "methods": list(METHOD_NAMES),
        "query_count": total_queries,
        "candidate_count": len(all_method_hidden),
        "queries": query_summaries,
    }

    return all_method_hidden, all_audit, summary


def write_pool_outputs(
    method_hidden_candidates: list[dict],
    audit_candidates: list[dict],
    summary: Mapping[str, Any],
    *,
    judgments_path: str | Path,
    audit_path: str | Path,
    summary_path: str | Path,
    overwrite: bool = False,
) -> None:
    """Write method-hidden judgments, private audit data, and summary."""
    judgments_path = Path(judgments_path)
    audit_path = Path(audit_path)
    summary_path = Path(summary_path)

    if judgments_path.exists() and not overwrite:
        raise FileExistsError(
            f"{judgments_path} already exists. "
            "Use --overwrite only before manual grading."
        )

    for path in (
        judgments_path,
        audit_path,
        summary_path,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    fieldnames = [
        "query_id",
        "split",
        "query",
        "paper_id",
        "title",
        "abstract",
        "year",
        "category",
        "grade",
        "notes",
    ]

    with judgments_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(method_hidden_candidates)

    with audit_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in audit_candidates:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pool method-hidden candidates from BM25, vector, "
            "graph and hybrid retrieval"
        )
    )
    parser.add_argument(
        "--benchmark",
        default=(
            "evaluation/benchmarks/"
            "retrieval_queries.json"
        ),
    )
    parser.add_argument(
        "--pool-depth",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--judgments-output",
        default=(
            "evaluation/benchmarks/"
            "retrieval_judgments.csv"
        ),
    )
    parser.add_argument(
        "--audit-output",
        default=(
            "results/retrieval/pooling/"
            "pooling_audit.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "results/retrieval/pooling/"
            "pooling_summary.json"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an ungraded judgment file. "
            "Never use after manual grading starts."
        ),
    )
    arguments = parser.parse_args()

    benchmark = load_benchmark(
        arguments.benchmark,
        require_judgments=False,
    )

    judgments_path = Path(
        arguments.judgments_output
    )
    if judgments_path.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"{judgments_path} already exists. "
            "Refusing to overwrite possible human judgments."
        )

    driver = get_driver()

    try:
        retrievers = build_retrievers(driver)
        method_hidden, audit, summary = pool_benchmark(
            benchmark,
            retrievers,
            pool_depth=arguments.pool_depth,
        )
    finally:
        driver.close()

    write_pool_outputs(
        method_hidden,
        audit,
        summary,
        judgments_path=arguments.judgments_output,
        audit_path=arguments.audit_output,
        summary_path=arguments.summary_output,
        overwrite=arguments.overwrite,
    )

    print()
    print("Candidate pooling complete")
    print(
        f"Queries: {summary['query_count']}"
    )
    print(
        f"Method-hidden candidates: "
        f"{summary['candidate_count']}"
    )
    print(
        f"Judgment file: {arguments.judgments_output}"
    )
    print(
        f"Hidden audit file: {arguments.audit_output}"
    )
    print(
        f"Summary file: {arguments.summary_output}"
    )


if __name__ == "__main__":
    main()
