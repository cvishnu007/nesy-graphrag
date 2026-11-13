import numpy as np

from src.evaluation.contradiction_verdict_eval import auroc, best_threshold, binary_report, calibration, llm_report


def test_binary_report_and_auroc():
    report = binary_report([True, True, False, False], [True, False, True, False])
    assert report["precision"] == 0.5 and report["recall"] == 0.5 and report["accuracy"] == 0.5
    assert auroc([True, False], [0.9, 0.1]) == 1.0 and auroc([True, False], [0.1, 0.9]) == 0.0
    assert auroc([True, True], [0.5, 0.6]) is None


def test_threshold_is_chosen_from_scores():
    assert best_threshold([True, True, False], np.array([0.8, 0.7, 0.2])) == 0.7


def test_overconfident_wrong_answers_raise_calibration_error():
    calibrated = calibration([True, False], [True, False], [1.0, 1.0])
    overconfident = calibration([True, False], [False, True], [1.0, 1.0])
    assert calibrated["expected_calibration_error"] == 0.0
    assert overconfident["expected_calibration_error"] == 1.0


def test_llm_report_counts_invalid_outputs_as_non_contradictions():
    items = [{"gold": True}, {"gold": False}]
    rows = [{"verdict": "UNKNOWN", "confidence": None, "valid": False},
            {"verdict": "AGREEMENT", "confidence": 0.9, "valid": True}]
    report = llm_report(items, rows)
    assert report["valid_rate"] == 0.5 and report["binary"]["recall"] == 0.0


def test_rag_answer_quality_scores_support_per_arm():
    from src.evaluation.rag_answer_quality import score_reviews

    def provider(claim, passage):
        return {"label": "SUPPORTED" if "true" in claim else "CONTRADICTED", "confidence": 0.9}

    passage = {"id": "P1", "text": "evidence"}
    rows = [
        {"method": "vector_rag", "query_id": "Q1", "passages": [passage],
         "provenance": {"claims": [{"text": "true claim", "cited_passage_ids": ["P1"]},
                                   {"text": "bad claim", "cited_passage_ids": ["P1"]}],
                        "stats": {"citation_precision": 1.0, "total_citations": 2}}},
        {"method": "graph_rag", "query_id": "Q1", "passages": [],
         "provenance": {"claims": [], "stats": {"citation_precision": 0.0, "total_citations": 0}}},
    ]
    summary, per_query = score_reviews(rows, provider)
    assert summary["vector_rag"]["semantic_support_rate"] == 0.5
    assert summary["vector_rag"]["contradicted_rate"] == 0.5
    assert summary["graph_rag"]["abstention_rate"] == 1.0
    assert per_query["Q1"] == {"vector_rag": 0.5, "graph_rag": 0.0}


def test_failed_requests_are_retried_on_resume(tmp_path):
    import json as _json

    from src.evaluation.contradiction_verdict_eval import run_llm

    checkpoint = tmp_path / "ckpt.jsonl"
    checkpoint.write_text("\n".join(_json.dumps(row) for row in [
        {"item_id": "a", "verdict": "AGREEMENT", "confidence": 0.9, "valid": True, "raw": "{}"},
        {"item_id": "b", "verdict": "UNKNOWN", "confidence": None, "valid": False, "raw": "ERROR RateLimitError"},
    ]) + "\n")

    class Client:
        provider, model, calls = "groq", "m", []

        def complete(self, prompt, **kwargs):
            Client.calls.append(prompt)
            return '{"verdict": "CONTRADICTION", "confidence": 0.8, "reason": "r", "claim_1": "x", "claim_2": "y"}'

    rows = run_llm(Client(), [{"item_id": "a", "claim": "c", "evidence": "e"},
                              {"item_id": "b", "claim": "c", "evidence": "e"}], checkpoint)
    assert len(Client.calls) == 1  # only the failed item is retried
    assert [r["verdict"] for r in rows] == ["AGREEMENT", "CONTRADICTION"]
