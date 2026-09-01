import pytest

from src.evaluation.claim_support_metrics import evaluate_claim_support
from src.evaluation.semantic_support import (
    aggregate_passage_decisions,
    parse_support_decision,
    verify_claim_support,
)


def provider_for(label, confidence=0.9):
    return lambda claim, passage: {
        "label": label, "confidence": confidence, "reason": "fixture",
        "model": "test-provider",
    }


@pytest.mark.parametrize(
    "label", ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
)
def test_all_support_labels(label):
    result = verify_claim_support(
        "claim", [{"id": "P1", "text": "evidence"}],
        provider=provider_for(label), model="test-provider",
    )
    assert result["support_label"] == label
    assert result["valid"] is True


def test_empty_evidence_and_missing_provider_are_explicit():
    empty = verify_claim_support("claim", [], provider=provider_for("SUPPORTED"))
    assert empty["valid"] is False
    assert "no valid evidence passages" in empty["audit"]["errors"]
    unavailable = verify_claim_support(
        "claim", [{"id": "P1", "text": "evidence"}], provider=None
    )
    assert unavailable["valid"] is False
    assert "provider is unavailable" in unavailable["audit"]["errors"][0]


def test_malformed_and_low_confidence_provider_decisions_are_invalid():
    malformed = parse_support_decision("bad", passage_id="P1", model="test")
    assert malformed["valid"] is False
    low = verify_claim_support(
        "claim", [{"id": "P1", "text": "evidence"}],
        provider=provider_for("SUPPORTED", 0.69), min_confidence=0.70,
    )
    assert low["valid"] is False
    assert low["support_label"] == "UNSUPPORTED"


def test_contradiction_dominates_multiple_passages_deterministically():
    labels = {"P1": "SUPPORTED", "P2": "CONTRADICTED"}
    provider = lambda claim, passage: {"label": labels[passage["id"]], "confidence": 0.9}
    passages = [{"id": "P1", "text": "supports"}, {"id": "P2", "text": "conflicts"}]
    first = verify_claim_support("claim", passages, provider=provider)
    second = verify_claim_support("claim", reversed(passages), provider=provider)
    assert first["support_label"] == second["support_label"] == "CONTRADICTED"


def test_aggregation_rejects_invalid_confidence_threshold():
    with pytest.raises(ValueError):
        aggregate_passage_decisions([], min_confidence=1.1)


def test_claim_support_metrics_include_false_acceptance_and_coverage():
    result = evaluate_claim_support([
        {"label": "SUPPORTED", "prediction": "SUPPORTED", "valid": True},
        {"label": "UNSUPPORTED", "prediction": "SUPPORTED", "valid": True},
        {"label": "CONTRADICTED", "prediction": "UNKNOWN", "valid": False},
    ])
    assert result["false_acceptance_rate"] == 0.5
    assert result["unsupported_claim_rejection_rate"] == 0.5
    assert result["coverage"] == pytest.approx(2 / 3)
