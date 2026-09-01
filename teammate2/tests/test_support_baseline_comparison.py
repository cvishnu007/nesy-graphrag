import pytest

from src.evaluation.support_baseline_comparison import (
    compare_support_baselines,
    existence_only_predictions,
)


def item(item_id, label):
    return {
        "item_id": item_id, "passage_id": f"P-{item_id}",
        "passage_text": "evidence", "claim": "claim", "label": label,
    }


def test_existence_only_accepts_every_structurally_valid_item():
    rows = existence_only_predictions([
        item("S1", "SUPPORTED"), item("S2", "UNSUPPORTED")
    ])

    assert {row["prediction"] for row in rows} == {"SUPPORTED"}
    assert all(row["valid"] for row in rows)
    assert all(row["baseline"] == "passage_id_existence_only" for row in rows)


def test_comparison_reports_paired_metric_deltas():
    items = [item("S1", "SUPPORTED"), item("S2", "UNSUPPORTED")]
    semantic = [
        {"item_id": "S1", "prediction": "SUPPORTED", "confidence": 0.9, "valid": True},
        {"item_id": "S2", "prediction": "UNSUPPORTED", "confidence": 0.8, "valid": True},
    ]

    result = compare_support_baselines(items, semantic, threshold=0.7)

    assert result["semantic_minus_existence"]["macro_f1"] == pytest.approx(
        result["semantic"]["macro_f1"] - result["existence_only"]["macro_f1"]
    )
    assert result["reference_annotation_source"] == "AI-generated"
    assert result["human_ground_truth"] is False


def test_comparison_rejects_missing_or_extra_prediction_ids():
    items = [item("S1", "SUPPORTED")]
    with pytest.raises(ValueError, match="IDs differ"):
        compare_support_baselines(items, [], threshold=0.7)
    with pytest.raises(ValueError, match="IDs differ"):
        compare_support_baselines(
            items,
            [{"item_id": "S2", "prediction": "SUPPORTED", "confidence": 1.0}],
            threshold=0.7,
        )
