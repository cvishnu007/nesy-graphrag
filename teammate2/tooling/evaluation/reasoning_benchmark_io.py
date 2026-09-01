"""Strict loaders and validators for Teammate 2 reasoning benchmarks."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


VALID_SPLITS = {"train", "dev", "test"}
CONTRADICTION_LABELS = {
    "CONTRADICTION", "AGREEMENT", "DIFFERENT SCOPE", "UNCERTAIN"
}
SUPPORT_LABELS = {
    "SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"
}
HYPOTHESIS_DIMENSIONS = (
    "evidence", "novelty", "feasibility", "specificity", "usefulness"
)
HYPOTHESIS_SCORES = {1, 3, 5}


class ReasoningBenchmarkValidationError(ValueError):
    """Raised when a reasoning benchmark violates its declared schema."""


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReasoningBenchmarkValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _text(record: Mapping[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ReasoningBenchmarkValidationError(
            f"{context}.{field} must be a non-empty string"
        )
    return value.strip()


def _split(record: Mapping[str, Any], context: str) -> str:
    value = _text(record, "split", context)
    if value not in VALID_SPLITS:
        raise ReasoningBenchmarkValidationError(
            f"{context}.split must be one of {sorted(VALID_SPLITS)}"
        )
    return value


def _container(data: Any, collection: str) -> list[Mapping[str, Any]]:
    if not isinstance(data, Mapping):
        raise ReasoningBenchmarkValidationError("Benchmark must be a JSON object")
    _text(data, "benchmark_version", "benchmark")
    status = _text(data, "status", "benchmark")
    if status not in {"draft", "frozen"}:
        raise ReasoningBenchmarkValidationError(
            "benchmark.status must be 'draft' or 'frozen'"
        )
    if "fixture_only" in data and not isinstance(data["fixture_only"], bool):
        raise ReasoningBenchmarkValidationError("benchmark.fixture_only must be boolean")
    records = data.get(collection)
    if not isinstance(records, list):
        raise ReasoningBenchmarkValidationError(
            f"benchmark.{collection} must be a list"
        )
    if status == "frozen" and not records:
        raise ReasoningBenchmarkValidationError(
            f"A frozen benchmark cannot have an empty {collection} list"
        )
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ReasoningBenchmarkValidationError(
                f"{collection}[{index}] must be an object"
            )
    return records


def _check_reference(value: str, valid_values: set[str] | None, context: str) -> None:
    if valid_values is not None and value not in valid_values:
        raise ReasoningBenchmarkValidationError(f"Unknown reference {context}: {value}")


def validate_contradiction_benchmark(
    data: Mapping[str, Any], *, valid_paper_ids: set[str] | None = None
) -> None:
    pairs = _container(data, "pairs")
    pair_ids: set[str] = set()
    unordered_pairs: set[tuple[str, str]] = set()
    for index, record in enumerate(pairs):
        context = f"pairs[{index}]"
        pair_id = _text(record, "pair_id", context)
        _split(record, context)
        paper1 = _text(record, "paper1_id", context)
        paper2 = _text(record, "paper2_id", context)
        label = _text(record, "label", context)
        _text(record, "reason", context)
        if pair_id in pair_ids:
            raise ReasoningBenchmarkValidationError(f"Duplicate pair_id: {pair_id}")
        pair_ids.add(pair_id)
        if paper1 == paper2:
            raise ReasoningBenchmarkValidationError(f"{context} repeats the same paper")
        canonical = tuple(sorted((paper1, paper2)))
        if (paper1, paper2) != canonical:
            raise ReasoningBenchmarkValidationError(
                f"{context} paper IDs are not in canonical lexical order"
            )
        if canonical in unordered_pairs:
            raise ReasoningBenchmarkValidationError(
                f"Duplicate or reversed contradiction pair: {canonical}"
            )
        unordered_pairs.add(canonical)
        if label not in CONTRADICTION_LABELS:
            raise ReasoningBenchmarkValidationError(f"Invalid contradiction label: {label}")
        annotators = record.get("annotators")
        if not isinstance(annotators, list) or any(
            not isinstance(item, str) or not item.strip() for item in annotators
        ) or len(set(annotators)) != len(annotators):
            raise ReasoningBenchmarkValidationError(
                f"{context}.annotators must contain unique non-empty reviewer IDs"
            )
        if not isinstance(record.get("adjudicated"), bool):
            raise ReasoningBenchmarkValidationError(
                f"{context}.adjudicated must be boolean"
            )
        _check_reference(paper1, valid_paper_ids, f"{context}.paper1_id")
        _check_reference(paper2, valid_paper_ids, f"{context}.paper2_id")


def validate_claim_support_benchmark(
    data: Mapping[str, Any], *, valid_query_ids: set[str] | None = None,
    valid_paper_ids: set[str] | None = None,
    valid_passage_ids: set[str] | None = None,
) -> None:
    items = _container(data, "items")
    item_ids: set[str] = set()
    claim_passages: set[tuple[str, str]] = set()
    for index, record in enumerate(items):
        context = f"items[{index}]"
        item_id = _text(record, "item_id", context)
        _split(record, context)
        query_id = _text(record, "query_id", context)
        claim = _text(record, "claim", context)
        passage_id = _text(record, "passage_id", context)
        _text(record, "passage_text", context)
        paper_id = _text(record, "paper_id", context)
        label = _text(record, "label", context)
        if "notes" not in record or not isinstance(record["notes"], str):
            raise ReasoningBenchmarkValidationError(f"{context}.notes must be a string")
        if item_id in item_ids:
            raise ReasoningBenchmarkValidationError(f"Duplicate item_id: {item_id}")
        item_ids.add(item_id)
        identity = (claim.casefold().strip(), passage_id)
        if identity in claim_passages:
            raise ReasoningBenchmarkValidationError(
                f"Duplicate claim/passage item: {item_id}"
            )
        claim_passages.add(identity)
        if label not in SUPPORT_LABELS:
            raise ReasoningBenchmarkValidationError(f"Invalid support label: {label}")
        _check_reference(query_id, valid_query_ids, f"{context}.query_id")
        _check_reference(paper_id, valid_paper_ids, f"{context}.paper_id")
        _check_reference(passage_id, valid_passage_ids, f"{context}.passage_id")


def _validate_rating(rating: Mapping[str, Any], context: str) -> None:
    _text(rating, "reviewer_id", context)
    for dimension in HYPOTHESIS_DIMENSIONS:
        score = rating.get(dimension)
        if isinstance(score, bool) or score not in HYPOTHESIS_SCORES:
            raise ReasoningBenchmarkValidationError(
                f"{context}.{dimension} must be one of {sorted(HYPOTHESIS_SCORES)}"
            )
    if "notes" in rating and not isinstance(rating["notes"], str):
        raise ReasoningBenchmarkValidationError(f"{context}.notes must be a string")


def validate_hypothesis_benchmark(
    data: Mapping[str, Any], *, valid_query_ids: set[str] | None = None
) -> None:
    hypotheses = _container(data, "hypotheses")
    hypothesis_ids: set[str] = set()
    for index, record in enumerate(hypotheses):
        context = f"hypotheses[{index}]"
        hypothesis_id = _text(record, "hypothesis_id", context)
        _split(record, context)
        query_id = _text(record, "query_id", context)
        _text(record, "hypothesis", context)
        if hypothesis_id in hypothesis_ids:
            raise ReasoningBenchmarkValidationError(
                f"Duplicate hypothesis_id: {hypothesis_id}"
            )
        hypothesis_ids.add(hypothesis_id)
        ratings = record.get("ratings")
        if not isinstance(ratings, list):
            raise ReasoningBenchmarkValidationError(f"{context}.ratings must be a list")
        reviewers: set[str] = set()
        for rating_index, rating in enumerate(ratings):
            rating_context = f"{context}.ratings[{rating_index}]"
            if not isinstance(rating, Mapping):
                raise ReasoningBenchmarkValidationError(f"{rating_context} must be an object")
            _validate_rating(rating, rating_context)
            reviewer = rating["reviewer_id"].strip()
            if reviewer in reviewers:
                raise ReasoningBenchmarkValidationError(
                    f"Duplicate reviewer {reviewer} for {hypothesis_id}"
                )
            reviewers.add(reviewer)
        adjudication = record.get("adjudication")
        if adjudication is not None:
            if not isinstance(adjudication, Mapping):
                raise ReasoningBenchmarkValidationError(
                    f"{context}.adjudication must be an object"
                )
            _validate_rating(adjudication, f"{context}.adjudication")
        model_feasibility = record.get("model_feasibility")
        if model_feasibility is not None and model_feasibility not in {"HIGH", "MEDIUM", "LOW"}:
            raise ReasoningBenchmarkValidationError(
                f"{context}.model_feasibility must be HIGH, MEDIUM, or LOW"
            )
        hns = record.get("hns")
        if hns is not None and (isinstance(hns, bool) or not isinstance(hns, (int, float)) or not 0 <= hns <= 1):
            raise ReasoningBenchmarkValidationError(f"{context}.hns must be in [0, 1]")
        _check_reference(query_id, valid_query_ids, f"{context}.query_id")


VALIDATORS = {
    "contradiction": validate_contradiction_benchmark,
    "support": validate_claim_support_benchmark,
    "hypothesis": validate_hypothesis_benchmark,
}


def load_reasoning_benchmark(path: str | Path, task: str) -> dict:
    """Load JSON with duplicate-key protection, then validate it for ``task``."""
    if task not in VALIDATORS:
        raise ReasoningBenchmarkValidationError(f"Unknown reasoning task: {task}")
    try:
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file, object_pairs_hook=_reject_duplicate_json_keys)
    except ReasoningBenchmarkValidationError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ReasoningBenchmarkValidationError(
            f"Could not load reasoning benchmark {path}: {error}"
        ) from error
    VALIDATORS[task](data)
    return data


def records_for_split(data: Mapping[str, Any], task: str, split: str) -> list[dict]:
    if split not in VALID_SPLITS:
        raise ReasoningBenchmarkValidationError(f"Invalid split: {split}")
    collection = {"contradiction": "pairs", "support": "items", "hypothesis": "hypotheses"}.get(task)
    if collection is None:
        raise ReasoningBenchmarkValidationError(f"Unknown reasoning task: {task}")
    return [dict(item) for item in data[collection] if item["split"] == split]
