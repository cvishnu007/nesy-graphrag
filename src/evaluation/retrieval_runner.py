"""Run all retrieval methods against a judged benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

from src.evaluation.benchmark_io import load_benchmark
from src.evaluation.candidate_pool import METHOD_NAMES, build_retrievers
from src.evaluation.ir_metrics import aggregate_query_metrics, evaluate_ranking
from src.storage.neo4j_store import get_driver


K_VALUES = (5, 10, 20)


def _ranked_ids(results: list[Mapping[str, Any]], method: str) -> list[str]:
    ranked = []
    seen = set()
    for rank, paper in enumerate(results, start=1):
        if not isinstance(paper, Mapping):
            raise RuntimeError(f"{method} result at rank {rank} is not an object")
        paper_id = str(paper.get("id", "")).strip()
        if not paper_id:
            raise RuntimeError(f"{method} returned a blank paper ID at rank {rank}")
        if paper_id in seen:
            raise RuntimeError(f"{method} returned duplicate paper ID {paper_id}")
        seen.add(paper_id)
        ranked.append(paper_id)
    return ranked


def evaluate_benchmark(
    benchmark: Mapping[str, Any],
    retrievers: Mapping[str, Callable],
    *,
    split: str = "all",
    top_k: int = 20,
) -> tuple[list[dict], list[dict], dict]:
    """Evaluate the three NeSy ablation retrievers."""
    if split not in {"all", "dev", "test"}:
        raise ValueError("split must be all, dev, or test")
    if top_k < max(K_VALUES):
        raise ValueError(f"top_k must be at least {max(K_VALUES)}")
    missing = [method for method in METHOD_NAMES if method not in retrievers]
    if missing:
        raise ValueError(f"Missing retrievers: {', '.join(missing)}")

    queries = [
        query
        for query in benchmark["queries"]
        if split == "all" or query["split"] == split
    ]
    if not queries:
        raise ValueError(f"No benchmark queries found for split {split}")

    metric_rows: list[dict] = []
    ranking_rows: list[dict] = []
    for query_index, query in enumerate(queries, start=1):
        query_id = query["query_id"]
        query_text = query["query"]
        judgments = query["judgments"]
        print(f"[{query_index}/{len(queries)}] {query_id}: {query_text}")

        for method in METHOD_NAMES:
            started = perf_counter()
            try:
                results = list(retrievers[method](query_text, top_k))
            except Exception as error:
                raise RuntimeError(
                    f"Evaluation failed for {query_id} using {method}: {error}"
                ) from error
            elapsed_ms = (perf_counter() - started) * 1000
            ranked_ids = _ranked_ids(results, method)
            metrics = evaluate_ranking(ranked_ids, judgments, K_VALUES)
            row = {
                "query_id": query_id,
                "split": query["split"],
                "query": query_text,
                "method": method,
                "result_count": len(ranked_ids),
                "latency_ms": round(elapsed_ms, 3),
                **metrics,
            }
            metric_rows.append(row)
            ranking_rows.append(
                {
                    "query_id": query_id,
                    "split": query["split"],
                    "method": method,
                    "ranked_paper_ids": ranked_ids,
                }
            )
            print(
                f"  {method:<6} NDCG@10={metrics['ndcg@10']:.4f} "
                f"Recall@10={metrics['recall@10']:.4f} "
                f"Unjudged@20={metrics['unjudged_rate@20']:.4f}"
            )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in metric_rows:
        grouped[row["method"]].append(row)
    methods = {
        method: aggregate_query_metrics(grouped[method])
        for method in METHOD_NAMES
    }
    winner = max(
        METHOD_NAMES,
        key=lambda method: (
            methods[method]["metrics"]["ndcg@10"]["mean"],
            methods[method]["metrics"]["recall@10"]["mean"],
            method,
        ),
    )
    summary = {
        "benchmark_version": benchmark["benchmark_version"],
        "benchmark_status": benchmark.get("status", ""),
        "split": split,
        "top_k": top_k,
        "query_count": len(queries),
        "primary_metric": benchmark.get("primary_metric", "ndcg@10"),
        "secondary_metric": benchmark.get("secondary_metric", "recall@10"),
        "methods": methods,
        "winner_by_ndcg_at_10": winner,
    }
    return metric_rows, ranking_rows, summary


def write_outputs(
    metric_rows: list[dict],
    ranking_rows: list[dict],
    summary: dict,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write per-query CSV, rankings JSONL, and aggregate JSON."""
    directory = Path(output_dir)
    paths = {
        "metrics": directory / "per_query_metrics.csv",
        "rankings": directory / "rankings.jsonl",
        "summary": directory / "summary.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite evaluation outputs: " + ", ".join(existing)
        )
    directory.mkdir(parents=True, exist_ok=True)

    with paths["metrics"].open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with paths["rankings"].open("w", encoding="utf-8") as file:
        for row in ranking_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        default="evaluation/benchmarks/retrieval_queries_judged.json",
    )
    parser.add_argument("--split", choices=("all", "dev", "test"), default="all")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", default="results/retrieval/evaluation")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    benchmark = load_benchmark(arguments.benchmark, require_judgments=True)
    driver = get_driver()
    try:
        retrievers = build_retrievers(driver)
        rows, rankings, summary = evaluate_benchmark(
            benchmark,
            retrievers,
            split=arguments.split,
            top_k=arguments.top_k,
        )
        write_outputs(
            rows,
            rankings,
            summary,
            arguments.output_dir,
            overwrite=arguments.overwrite,
        )
    finally:
        driver.close()

    print("\nEvaluation complete")
    print(f"Winner by NDCG@10: {summary['winner_by_ndcg_at_10']}")
    print(f"Results: {arguments.output_dir}")


if __name__ == "__main__":
    main()
