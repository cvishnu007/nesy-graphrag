import pytest

from src.pipeline.metrics import compute_hns
from src.pipeline.hypothesis import (
    parse_hypothesis_response,
    partition_validated_hypotheses,
    score_hypothesis_candidate,
)


class PathResult:
    def __init__(self, path_length):
        self.path_length = path_length

    def single(self):
        return {"pathLen": self.path_length} if self.path_length is not None else None


class PathSession:
    def __init__(self, path_lengths):
        self.path_lengths = path_lengths

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def run(self, query, *, hid, qids):
        return PathResult(self.path_lengths.get(hid))


class PathDriver:
    def __init__(self, path_lengths):
        self.path_lengths = path_lengths

    def session(self):
        return PathSession(self.path_lengths)


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


def test_hns_rewards_longer_measured_graph_paths():
    driver = PathDriver({"near": 2, "far": 6})
    hypotheses = [{"paper": {"id": "near"}}, {"paper": {"id": "far"}}]

    result = compute_hns(driver, hypotheses, ["query-paper"])

    assert result["individual_scores"] == pytest.approx([1 / 3, 1.0], abs=0.0001)
    assert result["hns"] == pytest.approx(2 / 3, abs=0.0001)
