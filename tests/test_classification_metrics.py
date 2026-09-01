import pytest

from src.evaluation.classification_metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    f1,
    macro_f1,
    macro_precision,
    macro_recall,
    macro_metrics,
    precision,
    precision_recall_f1,
    recall,
)
from src.evaluation.contradiction_runner import (
    candidate_recall_at_k,
    evaluate_contradictions,
    pair_coverage,
    sweep_thresholds,
)


CLASSES = ["YES", "NO"]


def test_known_confusion_matrix_and_metrics():
    labels = ["YES", "YES", "NO", "NO"]
    predictions = ["YES", "NO", "YES", "NO"]
    assert confusion_matrix(labels, predictions, CLASSES) == [[1, 1], [1, 1]]
    assert accuracy(labels, predictions, CLASSES) == 0.5
    assert precision_recall_f1(labels, predictions, "YES", CLASSES)["f1"] == 0.5
    assert precision(labels, predictions, "YES", CLASSES) == 0.5
    assert recall(labels, predictions, "YES", CLASSES) == 0.5
    assert f1(labels, predictions, "YES", CLASSES) == 0.5
    assert macro_metrics(labels, predictions, CLASSES)["macro_f1"] == 0.5
    assert macro_precision(labels, predictions, CLASSES) == 0.5
    assert macro_recall(labels, predictions, CLASSES) == 0.5
    assert macro_f1(labels, predictions, CLASSES) == 0.5


def test_empty_inputs_and_missing_class_have_explicit_zero_behavior():
    report = classification_report([], [], CLASSES)
    assert report["accuracy"] == 0.0
    assert report["confusion_matrix"] == [[0, 0], [0, 0]]
    result = precision_recall_f1(["NO"], ["NO"], "YES", CLASSES)
    assert result["precision"] == result["recall"] == result["f1"] == 0.0


@pytest.mark.parametrize(
    "labels,predictions,classes,error",
    [(["YES"], [], CLASSES, ValueError), (["MAYBE"], ["YES"], CLASSES, ValueError),
     ([1], ["YES"], CLASSES, TypeError), (["YES"], ["YES"], ["YES", "YES"], ValueError)],
)
def test_classification_type_and_value_validation(labels, predictions, classes, error):
    with pytest.raises(error):
        confusion_matrix(labels, predictions, classes)


def test_candidate_recall_and_coverage_are_unordered():
    gold = [("a", "b"), ("c", "d")]
    ranked = [("b", "a"), ("x", "y"), ("d", "c")]
    assert candidate_recall_at_k(gold, ranked, 1) == 0.5
    assert candidate_recall_at_k(gold, ranked, 3) == 1.0
    assert pair_coverage(gold, ranked) == 1.0


def test_duplicate_ranked_pair_is_rejected():
    with pytest.raises(ValueError, match="Duplicate unordered pair"):
        candidate_recall_at_k([("a", "b")], [("a", "b"), ("b", "a")], 2)


def contradiction_rows():
    return [
        {"paper1_id": "a", "paper2_id": "b", "label": "CONTRADICTION", "prediction": "CONTRADICTION", "confidence": 0.8},
        {"paper1_id": "c", "paper2_id": "d", "label": "AGREEMENT", "prediction": "CONTRADICTION", "confidence": 0.6},
        {"paper1_id": "e", "paper2_id": "f", "label": "DIFFERENT SCOPE", "prediction": "UNKNOWN", "confidence": None},
        {"paper1_id": "g", "paper2_id": "h", "label": "UNCERTAIN", "prediction": "AGREEMENT", "confidence": 0.9},
    ]


def test_contradiction_evaluation_reports_malformed_and_uncertain():
    result = evaluate_contradictions(contradiction_rows())
    assert result["contradiction_precision"] == 0.5
    assert result["contradiction_recall"] == 1.0
    assert result["malformed_count"] == 1
    assert result["uncertain_gold_count"] == 1
    assert result["coverage"] == pytest.approx(2 / 3)


def test_threshold_sweep_is_deterministic():
    first = sweep_thresholds(contradiction_rows(), [0.5, 0.7])
    second = sweep_thresholds(contradiction_rows(), [0.5, 0.7])
    assert first == second
    assert first["best_threshold"] in {0.5, 0.7}
