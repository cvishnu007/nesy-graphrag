import json

import pytest

from src.evaluation.artifact_manifest import (
    directory_fingerprint,
    inspect_corpus,
    inspect_query_benchmark,
)
from src.evaluation.collect_reasoning_outputs import collect_with_functions
from src.evaluation.reasoning_candidate_export import (
    CandidateExportError,
    export_claim_support_candidates,
    export_contradiction_candidates,
    export_hypothesis_candidates,
)


def test_corpus_and_directory_manifest_are_deterministic(tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([
        {"id": "p1", "year": 2020}, {"id": "p2", "year": 2025}
    ]), encoding="utf-8")
    first = inspect_corpus(corpus)
    second = inspect_corpus(corpus)
    assert first == second
    assert first["paper_count"] == first["unique_id_count"] == 2
    assert directory_fingerprint(tmp_path)["file_count"] == 1


def test_query_manifest_records_versions_and_splits(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps({
        "benchmark_version": "fixture", "status": "draft", "corpus": {"paper_count": 2},
        "queries": [{"query_id": "Q1", "split": "dev"}, {"query_id": "Q2", "split": "test"}],
    }), encoding="utf-8")
    result = inspect_query_benchmark(path)
    assert result["splits"] == {"dev": 1, "test": 1}
    assert result["benchmark_version"] == "fixture"


def test_collect_outputs_orders_queries_and_reports_sanitized_failures():
    queries = [
        {"query_id": "Q2", "split": "dev", "query": "second"},
        {"query_id": "Q1", "split": "dev", "query": "first"},
    ]
    functions = {
        "contradiction": lambda query, top_k: [{"paper1": {}, "paper2": {}}],
        "support": lambda query, top_k: (_ for _ in ()).throw(RuntimeError("secret detail")),
    }
    outputs, failures = collect_with_functions(
        queries, ["contradiction", "support"], functions, top_k=5
    )
    assert [item["query_id"] for item in outputs["contradiction"]] == ["Q1", "Q2"]
    assert len(failures) == 2
    assert all("secret detail" not in item["message"] for item in failures)


def test_collect_outputs_rejects_caught_llm_failures():
    queries = [{"query_id": "Q1", "split": "test", "query": "query"}]
    outputs, failures = collect_with_functions(
        queries,
        ["support", "hypothesis", "contradiction"],
        {
            "support": lambda query, top_k: {"raw_answer": "LLM call failed: private"},
            "hypothesis": lambda query, top_k: {
                "hypotheses": [],
                "rejected_hypotheses": [{"llm_hypothesis": "LLM call failed: private"}],
            },
            "contradiction": lambda query, top_k: {
                "contradictions": [{"llm_analysis": "LLM call failed: private"}],
            },
        },
        top_k=5,
    )
    assert all(not records for records in outputs.values())
    assert len(failures) == 3
    assert all("private" not in failure["message"] for failure in failures)


def test_export_contradictions_deduplicates_reversed_pairs_without_labels():
    results = [{
        "query_id": "Q1", "split": "dev", "query": "query",
        "contradictions": [
            {"paper1": {"id": "b", "title": "B", "abstract": "B"},
             "paper2": {"id": "a", "title": "A", "abstract": "A"},
             "candidate_score": 0.8},
            {"paper1": {"id": "a", "title": "A", "abstract": "A"},
             "paper2": {"id": "b", "title": "B", "abstract": "B"}},
        ],
    }]
    candidates = export_contradiction_candidates(results)
    assert len(candidates) == 1
    assert candidates[0]["paper1"]["id"] == "a"
    assert "label" not in candidates[0]


def test_generation_configuration_reaches_candidates_and_protected_sidecar():
    configuration = {
        "primary": "alternate-model", "fallback": "alternate-model",
        "configuration_label": "revised", "revised_configuration": True,
    }
    candidates = export_contradiction_candidates([{
        "query_id": "Q1", "split": "dev", "query": "query",
        "generation_configuration": configuration,
        "contradictions": [{
            "paper1": {"id": "a", "title": "A", "abstract": "A"},
            "paper2": {"id": "b", "title": "B", "abstract": "B"},
        }],
    }])
    assert candidates[0]["generation_configuration"] == configuration


def test_export_claim_support_adds_unlabeled_difficult_negative():
    results = [{
        "query_id": "Q1", "split": "dev",
        "passages": [
            {"id": "S1", "paper_id": "P1", "text": "cited"},
            {"id": "S2", "paper_id": "P2", "text": "other"},
        ],
        "claims": [{"text": "claim", "cited_passage_ids": ["S1"]}],
    }]
    candidates = export_claim_support_candidates(results, negatives_per_claim=1)
    assert {item["candidate_source"] for item in candidates} == {
        "cited_passage_candidate", "difficult_negative_candidate"
    }
    assert all("label" not in item for item in candidates)


def test_export_claim_support_includes_unsupported_generated_claims_and_provenance():
    results = [{
        "query_id": "Q1", "split": "dev",
        "papers": [
            {"id": "P1", "score": 0.9, "source": "both"},
            {"id": "P2", "score": 0.8, "source": "neural"},
        ],
        "passages": [
            {"id": "S1", "paper_id": "P1", "text": "graph evidence"},
            {"id": "S2", "paper_id": "P2", "text": "graph comparison evidence"},
        ],
        "claims": [],
        "unsupported_claims": [{
            "text": "graph claim", "cited_passage_ids": ["S1"],
            "rejection_reasons": ["invalid_passage_ids"],
            "invalid_passage_ids": ["BAD"],
        }],
    }]
    candidates = export_claim_support_candidates(results, negatives_per_claim=1)
    assert len(candidates) == 2
    assert all(candidate["claim_grounded"] is False for candidate in candidates)
    assert candidates[0]["rejection_reasons"] == ["invalid_passage_ids"]


def test_export_hypotheses_includes_accepted_and_rejected_without_human_rating():
    item = {
        "paper": {"id": "P1", "supporting_paper_ids": ["P2"], "shared_concept_names": ["graph"]},
        "llm_hypothesis": (
            "HYPOTHESIS: Combine A and B.\nFEASIBILITY: MEDIUM\n"
            "SUPPORTING EVIDENCE: graph\nMISSING EVIDENCE: experiment"
        ),
        "feasibility": "MEDIUM", "accepted": True,
    }
    rejected = {**item, "llm_hypothesis": item["llm_hypothesis"].replace("A and B", "C and D"), "accepted": False}
    candidates = export_hypothesis_candidates([{
        "query_id": "Q1", "split": "test", "hypotheses": [item],
        "rejected_hypotheses": [rejected],
    }])
    assert len(candidates) == 2
    assert all("ratings" not in candidate and "label" not in candidate for candidate in candidates)
    assert all(candidate["evidence"][0]["supporting_evidence"] == "graph" for candidate in candidates)
    assert all(candidate["raw_generation"] for candidate in candidates)


def test_rejected_malformed_hypothesis_uses_existing_graph_candidate_text():
    candidates = export_hypothesis_candidates([{
        "query_id": "Q1", "split": "test", "hypotheses": [],
        "rejected_hypotheses": [{
            "paper": {
                "id": "P1", "hypothesis": "Existing structural-hole candidate",
                "supporting_paper_ids": ["P2"],
            },
            "llm_hypothesis": "Malformed model response without required fields",
            "feasibility": "UNKNOWN", "accepted": False, "validation_valid": False,
        }],
    }])
    assert candidates[0]["hypothesis"] == "Existing structural-hole candidate"
    assert candidates[0]["accepted"] is False
    assert candidates[0]["raw_generation"].startswith("Malformed model response")


def test_candidate_export_rejects_missing_query_identity():
    with pytest.raises(CandidateExportError):
        export_contradiction_candidates([{"split": "dev", "contradictions": []}])
