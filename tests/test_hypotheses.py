import unittest

from src.pipeline.hypothesis import (
    parse_hypothesis_response,
    partition_validated_hypotheses,
    score_hypothesis_candidate,
)


class HypothesisValidationTests(unittest.TestCase):
    def test_candidate_score_combines_overlap_and_query_support(self):
        overlap, support_ratio, score = score_hypothesis_candidate(
            shared_concepts=3,
            candidate_concepts=6,
            supporting_papers=2,
            query_papers=5,
        )
        self.assertEqual(overlap, 0.5)
        self.assertEqual(support_ratio, 0.4)
        self.assertAlmostEqual(score, 0.46)

    def test_parses_structured_feasibility_and_evidence(self):
        parsed = parse_hypothesis_response(
            "HYPOTHESIS: Combine A and B.\n"
            "FEASIBILITY: MEDIUM\n"
            "SUPPORTING EVIDENCE: Shared concept X.\n"
            "MISSING EVIDENCE: Controlled experiment.\n"
            "RATIONALE: They may complement each other.\n"
            "POTENTIAL IMPACT: Better results."
        )
        self.assertEqual(parsed["feasibility"], "MEDIUM")
        self.assertEqual(parsed["missing_evidence"], "Controlled experiment.")
        self.assertTrue(parsed["valid"])

    def test_missing_feasibility_is_invalid(self):
        parsed = parse_hypothesis_response("HYPOTHESIS: An unstructured idea.")
        self.assertEqual(parsed["feasibility"], "UNKNOWN")
        self.assertFalse(parsed["valid"])

    def test_parses_markdown_and_multiline_evidence(self):
        parsed = parse_hypothesis_response(
            "**HYPOTHESIS:** Combine A and B.\n\n"
            "**FEASIBILITY:** HIGH\n\n"
            "**MISSING EVIDENCE:** A controlled trial.\n\n"
            "**SUPPORTING EVIDENCE:**\n- Paper A\n- Shared concept B\n\n"
            "**RATIONALE:** Useful combination.\n"
            "**POTENTIAL IMPACT:** Better outcomes."
        )
        self.assertEqual(parsed["feasibility"], "HIGH")
        self.assertIn("Paper A", parsed["supporting_evidence"])
        self.assertTrue(parsed["valid"])

    def test_rejected_hypotheses_are_kept_for_audit(self):
        accepted, rejected = partition_validated_hypotheses([
            {"id": "good", "accepted": True},
            {"id": "low", "accepted": False},
            {"id": "invalid"},
        ])
        self.assertEqual([item["id"] for item in accepted], ["good"])
        self.assertEqual([item["id"] for item in rejected], ["low", "invalid"])


if __name__ == "__main__":
    unittest.main()
