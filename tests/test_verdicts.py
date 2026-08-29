import pytest

from src.pipeline.contradiction import score_contradiction_candidate
from src.pipeline.verdicts import (
    contradiction_verdict,
    is_confident_contradiction,
    parse_contradiction_response,
)


def test_parses_exact_structured_verdict_and_confidence():
    parsed = parse_contradiction_response(
        "VERDICT: CONTRADICTION\nCONFIDENCE: 0.85\nREASON: Claims conflict."
    )

    assert parsed["verdict"] == "CONTRADICTION"
    assert parsed["confidence"] == 0.85
    assert parsed["valid"] is True


def test_does_not_treat_keyword_mentions_as_verdict():
    parsed = parse_contradiction_response(
        "VERDICT: DIFFERENT SCOPE\nCONFIDENCE: 0.9\n"
        "REASON: This is not a contradiction."
    )

    assert parsed["verdict"] == "DIFFERENT SCOPE"


@pytest.mark.parametrize(
    "response",
    ["The papers probably agree.", "", "CONFIDENCE: 0.90"],
)
def test_malformed_response_is_unknown(response):
    parsed = parse_contradiction_response(response)

    assert parsed["verdict"] == "UNKNOWN"
    assert parsed["valid"] is False


def test_structured_item_takes_precedence_over_legacy_analysis():
    verdict = contradiction_verdict(
        {
            "verdict": "AGREEMENT",
            "llm_analysis": "VERDICT: CONTRADICTION",
        }
    )

    assert verdict == "AGREEMENT"


def test_candidate_score_normalizes_concept_set_size():
    jaccard, year_gap, score = score_contradiction_candidate(
        shared_count=2,
        concepts1=5,
        concepts2=5,
        year1=2020,
        year2=2023,
    )

    assert jaccard == pytest.approx(0.25)
    assert year_gap == 3
    assert score == pytest.approx(0.3375)


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(0.69, False), (0.70, True), (1.0, True)],
)
def test_contradiction_requires_minimum_confidence(confidence, expected):
    item = {"verdict": "CONTRADICTION", "confidence": confidence}

    assert is_confident_contradiction(item, 0.70) is expected
