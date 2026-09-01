"""Offline contradiction benchmark evaluation built on existing verdict semantics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from src.evaluation.classification_metrics import classification_report


PRIMARY_CLASSES = ("CONTRADICTION", "AGREEMENT", "DIFFERENT SCOPE")
EVALUATION_CLASSES = (*PRIMARY_CLASSES, "UNKNOWN")


def canonical_pair(paper1_id: str, paper2_id: str) -> tuple[str, str]:
    if not isinstance(paper1_id, str) or not paper1_id.strip():
        raise ValueError("paper1_id must be a non-empty string")
    if not isinstance(paper2_id, str) or not paper2_id.strip():
        raise ValueError("paper2_id must be a non-empty string")
    if paper1_id == paper2_id:
        raise ValueError("A contradiction pair requires two different papers")
    return tuple(sorted((paper1_id, paper2_id)))


def _unique_pairs(pairs: Iterable[Sequence[str]]) -> list[tuple[str, str]]:
    result = []
    seen = set()
    for value in pairs:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
            raise TypeError("Each pair must contain exactly two paper IDs")
        pair = canonical_pair(value[0], value[1])
        if pair in seen:
            raise ValueError(f"Duplicate unordered pair: {pair}")
        seen.add(pair)
        result.append(pair)
    return result


def candidate_recall_at_k(gold_pairs, ranked_pairs, k: int) -> float:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    gold = set(_unique_pairs(gold_pairs))
    ranked = _unique_pairs(ranked_pairs)
    if not gold:
        return 0.0
    return len(gold.intersection(ranked[:k])) / len(gold)


def pair_coverage(gold_pairs, evaluated_pairs) -> float:
    gold = set(_unique_pairs(gold_pairs))
    evaluated = set(_unique_pairs(evaluated_pairs))
    return len(gold.intersection(evaluated)) / len(gold) if gold else 0.0


def confidence_bins(records: Iterable[dict], bins=(0.0, 0.5, 0.7, 0.9, 1.0)) -> list[dict]:
    edges = list(bins)
    if len(edges) < 2 or edges[0] != 0.0 or edges[-1] != 1.0 or any(
        left >= right for left, right in zip(edges, edges[1:])
    ):
        raise ValueError("bins must increase from 0.0 to 1.0")
    output = []
    materialized = list(records)
    for index, (low, high) in enumerate(zip(edges, edges[1:])):
        selected = []
        for record in materialized:
            confidence = record.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                continue
            if low <= confidence <= high if index == len(edges) - 2 else low <= confidence < high:
                selected.append(record)
        correct = sum(item.get("label") == item.get("prediction") for item in selected)
        output.append({
            "lower": low, "upper": high, "count": len(selected),
            "accuracy": correct / len(selected) if selected else 0.0,
        })
    return output


def evaluate_contradictions(records: Iterable[dict], *, threshold: float | None = None) -> dict:
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1):
        raise ValueError("threshold must be in [0, 1]")
    rows = list(records)
    labels = []
    predictions = []
    uncertain = malformed = rejected = 0
    seen_pairs = set()
    for index, row in enumerate(rows):
        pair = canonical_pair(row.get("paper1_id"), row.get("paper2_id"))
        if pair in seen_pairs:
            raise ValueError(f"Duplicate unordered pair at record {index}: {pair}")
        seen_pairs.add(pair)
        label = row.get("label")
        if label == "UNCERTAIN":
            uncertain += 1
            continue
        if label not in PRIMARY_CLASSES:
            raise ValueError(f"Invalid gold contradiction label: {label}")
        prediction = row.get("prediction", "UNKNOWN")
        confidence = row.get("confidence")
        valid = prediction in PRIMARY_CLASSES and isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and 0 <= confidence <= 1
        if not valid:
            prediction = "UNKNOWN"
            malformed += 1
        elif threshold is not None and confidence < threshold:
            prediction = "UNKNOWN"
            rejected += 1
        labels.append(label)
        predictions.append(prediction)
    report = classification_report(labels, predictions, EVALUATION_CLASSES, macro_classes=PRIMARY_CLASSES)
    contradiction = report["per_class"]["CONTRADICTION"]
    report.update({
        "contradiction_precision": contradiction["precision"],
        "contradiction_recall": contradiction["recall"],
        "contradiction_f1": contradiction["f1"],
        "uncertain_gold_count": uncertain,
        "malformed_count": malformed,
        "rejected_count": rejected,
        "coverage": (len(labels) - malformed - rejected) / len(labels) if labels else 0.0,
        "confidence_bins": confidence_bins(rows),
    })
    return report


def sweep_thresholds(records: Iterable[dict], thresholds=None) -> dict:
    rows = list(records)
    values = list(thresholds) if thresholds is not None else [value / 100 for value in range(50, 91, 5)]
    if not values:
        raise ValueError("At least one threshold is required")
    runs = []
    for threshold in values:
        metrics = evaluate_contradictions(rows, threshold=threshold)
        runs.append({"threshold": threshold, "macro_f1": metrics["macro_f1"], "coverage": metrics["coverage"]})
    best = max(runs, key=lambda item: (item["macro_f1"], item["coverage"], -item["threshold"]))
    return {"selection_metric": "macro_f1", "best_threshold": best["threshold"], "runs": runs}
