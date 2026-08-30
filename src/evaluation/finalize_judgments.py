"""Merge a completed method-hidden judgment CSV into benchmark JSON."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.evaluation.benchmark_io import (
    BenchmarkValidationError,
    load_benchmark,
    validate_benchmark,
)


REQUIRED_COLUMNS = {
    "query_id",
    "split",
    "query",
    "paper_id",
    "grade",
    "notes",
}


def load_judgments(path: str | Path) -> dict[str, dict[str, int]]:
    """Load and strictly validate graded CSV rows."""
    judgments_path = Path(path)
    with judgments_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS.difference(columns)
        if missing:
            raise BenchmarkValidationError(
                f"Judgment CSV is missing columns: {sorted(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise BenchmarkValidationError("Judgment CSV is empty")

    result: dict[str, dict[str, int]] = defaultdict(dict)
    query_details: dict[str, tuple[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        query_id = row["query_id"].strip()
        paper_id = row["paper_id"].strip()
        grade_text = row["grade"].strip()
        notes = row["notes"].strip()
        if not query_id or not paper_id:
            raise BenchmarkValidationError(
                f"CSV row {row_number} has a blank query_id or paper_id"
            )
        if grade_text not in {"0", "1", "2"}:
            raise BenchmarkValidationError(
                f"CSV row {row_number} has invalid grade {grade_text!r}"
            )
        if not notes:
            raise BenchmarkValidationError(
                f"CSV row {row_number} has no judgment note"
            )
        if paper_id in result[query_id]:
            raise BenchmarkValidationError(
                f"Duplicate judgment for {query_id}/{paper_id}"
            )

        details = (row["split"].strip(), row["query"].strip())
        if query_id in query_details and query_details[query_id] != details:
            raise BenchmarkValidationError(
                f"Inconsistent query text or split for {query_id}"
            )
        query_details[query_id] = details
        result[query_id][paper_id] = int(grade_text)

    return dict(result)


def merge_judgments(
    benchmark: dict,
    judgments: dict[str, dict[str, int]],
) -> dict:
    """Return a new benchmark containing the supplied judgments."""
    merged = json.loads(json.dumps(benchmark))
    expected_ids = {query["query_id"] for query in merged["queries"]}
    supplied_ids = set(judgments)
    if supplied_ids != expected_ids:
        missing = sorted(expected_ids.difference(supplied_ids))
        extra = sorted(supplied_ids.difference(expected_ids))
        raise BenchmarkValidationError(
            f"Judgment query IDs differ; missing={missing}, extra={extra}"
        )

    for query in merged["queries"]:
        query["judgments"] = judgments[query["query_id"]]

    merged["benchmark_version"] = "0.2-draft"
    merged["status"] = "judgments_pending_human_review"
    merged["judgment_metadata"] = {
        "scale": {"0": "not relevant", "1": "partially relevant", "2": "directly relevant"},
        "source": "fast title-and-abstract relevance pass",
        "human_review_required": True,
    }
    validate_benchmark(merged, require_judgments=True)
    return merged


def write_benchmark(benchmark: dict, path: str | Path) -> None:
    """Atomically write formatted benchmark JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        default="evaluation/benchmarks/retrieval_queries.json",
    )
    parser.add_argument(
        "--judgments",
        default="evaluation/benchmarks/retrieval_judgments_draft.csv",
    )
    parser.add_argument(
        "--output",
        default="evaluation/benchmarks/retrieval_queries_judged.json",
    )
    arguments = parser.parse_args()

    benchmark = load_benchmark(arguments.benchmark)
    judgments = load_judgments(arguments.judgments)
    merged = merge_judgments(benchmark, judgments)
    write_benchmark(merged, arguments.output)

    total = sum(len(query["judgments"]) for query in merged["queries"])
    relevant = sum(
        grade > 0
        for query in merged["queries"]
        for grade in query["judgments"].values()
    )
    print(f"Judged benchmark saved: {arguments.output}")
    print(f"Queries: {len(merged['queries'])}")
    print(f"Judgments: {total}")
    print(f"Relevant (grade 1 or 2): {relevant}")


if __name__ == "__main__":
    main()
