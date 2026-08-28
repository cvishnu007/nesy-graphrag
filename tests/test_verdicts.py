import unittest

from src.pipeline.contradiction import score_contradiction_candidate
from src.pipeline.verdicts import (
    contradiction_verdict,
    is_confident_contradiction,
    parse_contradiction_response,
)


class ContradictionVerdictTests(unittest.TestCase):
    def test_parses_exact_structured_verdict_and_confidence(self):
        parsed = parse_contradiction_response(
            "VERDICT: CONTRADICTION\nCONFIDENCE: 0.85\nREASON: Claims conflict."
        )
        self.assertEqual(parsed["verdict"], "CONTRADICTION")
        self.assertEqual(parsed["confidence"], 0.85)
        self.assertTrue(parsed["valid"])

    def test_does_not_treat_keyword_mentions_as_verdict(self):
        parsed = parse_contradiction_response(
            "VERDICT: DIFFERENT SCOPE\nCONFIDENCE: 0.9\n"
            "REASON: This is not a contradiction."
        )
        self.assertEqual(parsed["verdict"], "DIFFERENT SCOPE")

    def test_malformed_response_is_unknown(self):
        parsed = parse_contradiction_response("The papers probably agree.")
        self.assertEqual(parsed["verdict"], "UNKNOWN")
        self.assertFalse(parsed["valid"])

    def test_structured_item_takes_precedence_over_legacy_analysis(self):
        verdict = contradiction_verdict({
            "verdict": "AGREEMENT",
            "llm_analysis": "VERDICT: CONTRADICTION",
        })
        self.assertEqual(verdict, "AGREEMENT")

    def test_candidate_score_normalizes_concept_set_size(self):
        jaccard, year_gap, score = score_contradiction_candidate(
            shared_count=2,
            concepts1=5,
            concepts2=5,
            year1=2020,
            year2=2023,
        )
        self.assertAlmostEqual(jaccard, 0.25)
        self.assertEqual(year_gap, 3)
        self.assertAlmostEqual(score, 0.3375)

    def test_contradiction_requires_minimum_confidence(self):
        low = {"verdict": "CONTRADICTION", "confidence": 0.69}
        high = {"verdict": "CONTRADICTION", "confidence": 0.70}

        self.assertFalse(is_confident_contradiction(low, 0.70))
        self.assertTrue(is_confident_contradiction(high, 0.70))


if __name__ == "__main__":
    unittest.main()
