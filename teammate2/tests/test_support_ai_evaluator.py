import json

import pytest

from src.evaluation.support_ai_evaluator import (
    SupportAIEvaluationError,
    build_support_prompt,
    parse_support_batch,
    nli_scores_to_prediction,
    select_confidence_threshold,
)


def fixture_item(**overrides):
    item = {
        "item_id": "S1",
        "split": "dev",
        "query_id": "Q1",
        "claim": "A claim",
        "passage_id": "P1",
        "passage_text": "Evidence text",
        "paper_id": "paper-1",
        "label": "UNSUPPORTED",
        "notes": "reference note",
        "annotators": ["ai-pass-1"],
        "adjudicated": False,
    }
    item.update(overrides)
    return item


def test_support_prompt_blinds_all_reference_fields():
    prompt = build_support_prompt([fixture_item()])

    assert "A claim" in prompt
    assert "Evidence text" in prompt
    assert '"label": "UNSUPPORTED"' not in prompt
    assert "reference note" not in prompt
    assert "annotators" not in prompt
    assert "adjudicated" not in prompt


def test_parse_support_batch_requires_exact_ids_and_valid_decisions():
    raw = json.dumps({
        "annotations": [{
            "id": "S1", "label": "SUPPORTED", "confidence": 0.91,
            "reason": "Direct support",
        }]
    })

    parsed = parse_support_batch(raw, ["S1"], model="test-model")

    assert parsed[0]["prediction"] == "SUPPORTED"
    assert parsed[0]["valid"] is True
    assert parsed[0]["human_review"] is False
    with pytest.raises(SupportAIEvaluationError):
        parse_support_batch(raw, ["S2"], model="test-model")


def test_threshold_selection_uses_only_supplied_development_rows():
    rows = [
        {"label": "SUPPORTED", "prediction": "SUPPORTED", "confidence": 0.9, "valid": True},
        {"label": "UNSUPPORTED", "prediction": "SUPPORTED", "confidence": 0.6, "valid": True},
    ]

    result = select_confidence_threshold(rows, thresholds=[0.0, 0.7])

    assert result["selected_threshold"] == 0.7
    assert len(result["candidates"]) == 2


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ({"contradiction": 0.8, "entailment": 0.1, "neutral": 0.1}, "CONTRADICTED"),
        ({"contradiction": 0.1, "entailment": 0.8, "neutral": 0.1}, "SUPPORTED"),
        ({"contradiction": 0.1, "entailment": 0.3, "neutral": 0.6}, "PARTIALLY_SUPPORTED"),
        ({"contradiction": 0.1, "entailment": 0.1, "neutral": 0.8}, "UNSUPPORTED"),
    ],
)
def test_local_nli_mapping_is_explicit(scores, expected):
    result = nli_scores_to_prediction(scores, partial_entailment_floor=0.2)
    assert result["label"] == expected
    assert result["valid"] is True
