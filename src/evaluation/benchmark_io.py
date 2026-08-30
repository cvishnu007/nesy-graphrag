"""Loading and validation for the judged retrieval benchmark."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


VALID_SPLITS = {"dev", "test"}
VALID_GRADES = {0, 1, 2}
MIN_QUERY_COUNT = 20
MAX_QUERY_COUNT = 50


class BenchmarkValidationError(ValueError):
    """Raised when the benchmark structure or values are invalid."""


def _reject_duplicate_json_keys(pairs):
    """Reject duplicate keys instead of silently overwriting them."""
    result = {}

    for key, value in pairs:
        if key in result:
            raise BenchmarkValidationError(
                f"Duplicate JSON key found: {key}"
            )
        result[key] = value

    return result


def _require_nonempty_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkValidationError(
            f"{field_name} must be a non-empty string"
        )


def validate_benchmark(
    benchmark: Mapping[str, Any],
    *,
    require_judgments: bool = False,
    valid_paper_ids: set[str] | None = None,
) -> None:
    """Validate benchmark metadata, queries, splits and judgments."""
    if not isinstance(benchmark, Mapping):
        raise BenchmarkValidationError(
            "Benchmark root must be a JSON object"
        )

    _require_nonempty_text(
        benchmark.get("benchmark_version"),
        "benchmark_version",
    )

    corpus = benchmark.get("corpus")
    if not isinstance(corpus, Mapping):
        raise BenchmarkValidationError(
            "corpus must be a JSON object"
        )

    for field in ("source", "years", "collection"):
        _require_nonempty_text(
            corpus.get(field),
            f"corpus.{field}",
        )

    paper_count = corpus.get("paper_count")
    if (
        isinstance(paper_count, bool)
        or not isinstance(paper_count, int)
        or paper_count <= 0
    ):
        raise BenchmarkValidationError(
            "corpus.paper_count must be a positive integer"
        )

    queries = benchmark.get("queries")
    if not isinstance(queries, list):
        raise BenchmarkValidationError(
            "queries must be a list"
        )

    if not MIN_QUERY_COUNT <= len(queries) <= MAX_QUERY_COUNT:
        raise BenchmarkValidationError(
            f"queries must contain between "
            f"{MIN_QUERY_COUNT} and {MAX_QUERY_COUNT} entries"
        )

    seen_ids = set()
    seen_texts = set()
    seen_splits = set()

    for index, item in enumerate(queries):
        location = f"queries[{index}]"

        if not isinstance(item, Mapping):
            raise BenchmarkValidationError(
                f"{location} must be a JSON object"
            )

        query_id = item.get("query_id")
        _require_nonempty_text(
            query_id,
            f"{location}.query_id",
        )

        if query_id in seen_ids:
            raise BenchmarkValidationError(
                f"Duplicate query_id found: {query_id}"
            )
        seen_ids.add(query_id)

        query_text = item.get("query")
        _require_nonempty_text(
            query_text,
            f"{location}.query",
        )

        normalized_text = " ".join(
            query_text.lower().split()
        )
        if normalized_text in seen_texts:
            raise BenchmarkValidationError(
                f"Duplicate query text found: {query_text}"
            )
        seen_texts.add(normalized_text)

        split = item.get("split")
        if split not in VALID_SPLITS:
            raise BenchmarkValidationError(
                f"{location}.split must be dev or test"
            )
        seen_splits.add(split)

        judgments = item.get("judgments")
        if not isinstance(judgments, Mapping):
            raise BenchmarkValidationError(
                f"{location}.judgments must be a JSON object"
            )

        if require_judgments and not judgments:
            raise BenchmarkValidationError(
                f"{location} has no relevance judgments"
            )

        relevant_count = 0

        for paper_id, grade in judgments.items():
            _require_nonempty_text(
                paper_id,
                f"{location}.judgments paper ID",
            )

            if isinstance(grade, bool) or grade not in VALID_GRADES:
                raise BenchmarkValidationError(
                    f"{location} paper {paper_id} has invalid "
                    f"grade {grade}; expected 0, 1 or 2"
                )

            if valid_paper_ids is not None and paper_id not in valid_paper_ids:
                raise BenchmarkValidationError(
                    f"{location} contains unknown paper ID: {paper_id}"
                )

            if grade > 0:
                relevant_count += 1

        if require_judgments and relevant_count == 0:
            raise BenchmarkValidationError(
                f"{location} has no relevant paper with grade 1 or 2"
            )

    if seen_splits != VALID_SPLITS:
        raise BenchmarkValidationError(
            "Benchmark must contain both dev and test queries"
        )


def load_benchmark(
    path: str | Path,
    *,
    require_judgments: bool = False,
    valid_paper_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Load a JSON benchmark and validate it."""
    benchmark_path = Path(path)

    try:
        with benchmark_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            benchmark = json.load(
                file,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
    except json.JSONDecodeError as error:
        raise BenchmarkValidationError(
            f"Invalid benchmark JSON: {error}"
        ) from error

    validate_benchmark(
        benchmark,
        require_judgments=require_judgments,
        valid_paper_ids=valid_paper_ids,
    )
    return benchmark


def queries_for_split(
    benchmark: Mapping[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    """Return queries belonging to dev or test."""
    if split not in VALID_SPLITS:
        raise BenchmarkValidationError(
            "split must be dev or test"
        )

    return [
        dict(query)
        for query in benchmark["queries"]
        if query["split"] == split
    ]