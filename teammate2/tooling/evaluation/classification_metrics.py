"""Dependency-free classification metrics with explicit edge-case behavior."""

from __future__ import annotations

from collections.abc import Sequence


def _validate(
    labels: Sequence[str], predictions: Sequence[str], class_names: Sequence[str]
) -> tuple[list[str], list[str], list[str]]:
    if isinstance(labels, (str, bytes)) or isinstance(predictions, (str, bytes)):
        raise TypeError("labels and predictions must be sequences, not strings")
    actual = list(labels)
    predicted = list(predictions)
    classes = list(class_names)
    if len(actual) != len(predicted):
        raise ValueError("labels and predictions must have equal length")
    if not classes or any(not isinstance(name, str) or not name for name in classes):
        raise ValueError("class_names must contain non-empty strings")
    if len(set(classes)) != len(classes):
        raise ValueError("class_names must be unique")
    allowed = set(classes)
    for value in actual + predicted:
        if not isinstance(value, str):
            raise TypeError("labels and predictions must contain strings")
        if value not in allowed:
            raise ValueError(f"Unknown class label: {value}")
    return actual, predicted, classes


def confusion_matrix(labels, predictions, class_names) -> list[list[int]]:
    """Return rows=gold and columns=prediction in ``class_names`` order."""
    actual, predicted, classes = _validate(labels, predictions, class_names)
    positions = {name: index for index, name in enumerate(classes)}
    matrix = [[0 for _ in classes] for _ in classes]
    for gold, guess in zip(actual, predicted):
        matrix[positions[gold]][positions[guess]] += 1
    return matrix


def accuracy(labels, predictions, class_names) -> float:
    actual, predicted, _ = _validate(labels, predictions, class_names)
    if not actual:
        return 0.0
    return sum(gold == guess for gold, guess in zip(actual, predicted)) / len(actual)


def precision_recall_f1(
    labels, predictions, positive_label: str, class_names
) -> dict[str, float | int]:
    actual, predicted, classes = _validate(labels, predictions, class_names)
    if positive_label not in classes:
        raise ValueError(f"positive_label is not in class_names: {positive_label}")
    tp = sum(gold == positive_label and guess == positive_label for gold, guess in zip(actual, predicted))
    fp = sum(gold != positive_label and guess == positive_label for gold, guess in zip(actual, predicted))
    fn = sum(gold == positive_label and guess != positive_label for gold, guess in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def precision(labels, predictions, positive_label: str, class_names) -> float:
    return precision_recall_f1(labels, predictions, positive_label, class_names)["precision"]


def recall(labels, predictions, positive_label: str, class_names) -> float:
    return precision_recall_f1(labels, predictions, positive_label, class_names)["recall"]


def f1(labels, predictions, positive_label: str, class_names) -> float:
    return precision_recall_f1(labels, predictions, positive_label, class_names)["f1"]


def per_class_metrics(labels, predictions, class_names) -> dict[str, dict]:
    actual, predicted, classes = _validate(labels, predictions, class_names)
    return {
        name: precision_recall_f1(actual, predicted, name, classes)
        for name in classes
    }


def macro_metrics(
    labels, predictions, class_names, *, include_classes: Sequence[str] | None = None
) -> dict[str, float]:
    actual, predicted, classes = _validate(labels, predictions, class_names)
    selected = list(include_classes) if include_classes is not None else classes
    if not selected or any(name not in classes for name in selected):
        raise ValueError("include_classes must be a non-empty subset of class_names")
    values = [precision_recall_f1(actual, predicted, name, classes) for name in selected]
    return {
        "macro_precision": sum(item["precision"] for item in values) / len(values),
        "macro_recall": sum(item["recall"] for item in values) / len(values),
        "macro_f1": sum(item["f1"] for item in values) / len(values),
    }


def macro_precision(labels, predictions, class_names) -> float:
    return macro_metrics(labels, predictions, class_names)["macro_precision"]


def macro_recall(labels, predictions, class_names) -> float:
    return macro_metrics(labels, predictions, class_names)["macro_recall"]


def macro_f1(labels, predictions, class_names) -> float:
    return macro_metrics(labels, predictions, class_names)["macro_f1"]


def classification_report(
    labels, predictions, class_names, *, macro_classes: Sequence[str] | None = None
) -> dict:
    actual, predicted, classes = _validate(labels, predictions, class_names)
    return {
        "count": len(actual),
        "class_names": classes,
        "accuracy": accuracy(actual, predicted, classes),
        "confusion_matrix": confusion_matrix(actual, predicted, classes),
        "per_class": per_class_metrics(actual, predicted, classes),
        **macro_metrics(actual, predicted, classes, include_classes=macro_classes),
    }
