"""Metrics for declared reference claim/passage support decisions."""

from __future__ import annotations

from collections.abc import Iterable

from src.evaluation.classification_metrics import classification_report
from src.evaluation.reasoning_benchmark_io import SUPPORT_LABELS


SUPPORT_CLASSES = ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED")
PREDICTION_CLASSES = (*SUPPORT_CLASSES, "UNKNOWN")


def evaluate_claim_support(records: Iterable[dict]) -> dict:
    rows = list(records)
    labels = []
    predictions = []
    confident = 0
    for record in rows:
        label = record.get("label")
        if label not in SUPPORT_LABELS:
            raise ValueError(f"Invalid gold support label: {label}")
        prediction = record.get("prediction", record.get("support_label", "UNKNOWN"))
        valid = record.get("valid", prediction in SUPPORT_CLASSES)
        if prediction not in SUPPORT_CLASSES or not valid:
            prediction = "UNKNOWN"
        else:
            confident += 1
        labels.append(label)
        predictions.append(prediction)

    report = classification_report(
        labels, predictions, PREDICTION_CLASSES, macro_classes=SUPPORT_CLASSES
    )
    supported_gold = sum(label == "SUPPORTED" for label in labels)
    unsupported_gold = sum(label in {"UNSUPPORTED", "CONTRADICTED"} for label in labels)
    accepted_unsupported = sum(
        label in {"UNSUPPORTED", "CONTRADICTED"} and prediction in {"SUPPORTED", "PARTIALLY_SUPPORTED"}
        for label, prediction in zip(labels, predictions)
    )
    rejected_unsupported = sum(
        label in {"UNSUPPORTED", "CONTRADICTED"} and prediction in {"UNSUPPORTED", "CONTRADICTED", "UNKNOWN"}
        for label, prediction in zip(labels, predictions)
    )
    predicted_supported = sum(prediction == "SUPPORTED" for prediction in predictions)
    report.update({
        "supported_claim_rate": predicted_supported / len(labels) if labels else 0.0,
        "supported_gold_count": supported_gold,
        "unsupported_claim_rejection_rate": rejected_unsupported / unsupported_gold if unsupported_gold else 0.0,
        "false_acceptance_rate": accepted_unsupported / unsupported_gold if unsupported_gold else 0.0,
        "coverage": confident / len(labels) if labels else 0.0,
    })
    return report
