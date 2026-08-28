import pytest

from src.pipeline.hypothesis import (
    parse_hypothesis_response,
    partition_validated_hypotheses,
    score_hypothesis_candidate,
)


def test_candidate_score_combines_overlap_and_query_support():
    overlap, support_ratio, score = score_hypothesis_candidate(
        shared_concepts=3,
        candidate_concepts=6,
        supporting_papers=2,
        query_papers=5,
    )

    assert overlap == 0.5
    assert support_ratio == 0.4
    assert score == pytest.approx(0.46)


def test_parses_structured_feasibility_and_evidence():
    parsed = parse_hypothesis_response(
        "HYPOTHESIS: Combine A and B.\n"
        "FEASIBILITY: MEDIUM\n"
        "SUPPORTING EVIDENCE: Shared concept X.\n"
        "MISSING EVIDENCE: Controlled experiment.\n"
        "RATIONALE: They may complement each other.\n"
        "POTENTIAL IMPACT: Better results."
    )

    assert parsed["feasibility"] == "MEDIUM"
    assert parsed["missing_evidence"] == "Controlled experiment."
    assert parsed["valid"] is True


@pytest.mark.parametrize(
    "response",
    [
        "HYPOTHESIS: An unstructured idea.",
        "HYPOTHESIS: Idea.\nFEASIBILITY: UNKNOWN",
    ],
)
def test_missing_or_unknown_feasibility_is_invalid(response):
    parsed = parse_hypothesis_response(response)

    assert parsed["feasibility"] == "UNKNOWN"
    assert parsed["valid"] is False


def test_parses_markdown_and_multiline_evidence():
    parsed = parse_hypothesis_response(
        "**HYPOTHESIS:** Combine A and B.\n\n"
        "**FEASIBILITY:** HIGH\n\n"
        "**MISSING EVIDENCE:** A controlled trial.\n\n"
        "**SUPPORTING EVIDENCE:**\n- Paper A\n- Shared concept B\n\n"
        "**RATIONALE:** Useful combination.\n"
        "**POTENTIAL IMPACT:** Better outcomes."
    )

    assert parsed["feasibility"] == "HIGH"
    assert "Paper A" in parsed["supporting_evidence"]
    assert parsed["valid"] is True


def test_rejected_hypotheses_are_kept_for_audit():
    accepted, rejected = partition_validated_hypotheses(
        [
            {"id": "good", "accepted": True},
            {"id": "low", "accepted": False},
            {"id": "invalid"},
        ]
    )

    assert [item["id"] for item in accepted] == ["good"]
    assert [item["id"] for item in rejected] == ["low", "invalid"]
