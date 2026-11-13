"""Tests for the November 2026 retrieval, graph, and reasoning improvements."""
import hashlib
import json

import numpy as np
import pytest

from src.evaluation.ir_metrics import condensed_ranking, evaluate_ranking
from src.evaluation.refresh_retrieval import _matches_digest
from src.evaluation.variant_ablation import score_variants
from src.pipeline.contradiction import score_contradiction_candidate
from src.pipeline.retrieval import _expansion_pattern, filter_symbolic_candidates, fuse_results
from src.pipeline.verdicts import contradiction_verdict, parse_contradiction_response
from src.storage.chroma_store import embedding_text
from src.storage.graph_enrichment import classify_concepts, generic_reason, top_k_neighbours


# -- graph enrichment -------------------------------------------------------

@pytest.mark.parametrize("name", ["this paper", "them", "these challenges", "the-art",
                                  "our approach", "recent years", "experimental results"])
def test_discourse_phrases_are_generic(name):
    assert generic_reason(name, 3, max_df=100) == "discourse_phrase"


@pytest.mark.parametrize("name", ["graph attention network", "federated learning", "sql injection"])
def test_scientific_terms_are_retained(name):
    assert generic_reason(name, 50, max_df=100) is None


def test_high_document_frequency_is_generic_and_reason_is_recorded():
    result = classify_concepts({"graph neural networks": 2000, "message passing": 40},
                               paper_count=10_000, max_df_fraction=0.01)
    assert result["graph neural networks"] == {"df": 2000, "generic": True, "reason": "document_frequency"}
    assert result["message passing"]["generic"] is False


def test_classify_rejects_invalid_settings():
    with pytest.raises(ValueError):
        classify_concepts({}, paper_count=0)
    with pytest.raises(ValueError):
        classify_concepts({}, paper_count=10, max_df_fraction=0)


def test_knn_excludes_self_and_ranks_by_cosine():
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    edges = top_k_neighbours(["a", "b", "c", "d"], vectors, k=1, block=2, device="cpu")
    pairs = {(edge["source"], edge["target"]) for edge in edges}
    assert pairs == {("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")}
    assert all(edge["rank"] == 1 and edge["source"] != edge["target"] for edge in edges)


def test_knn_validates_inputs():
    with pytest.raises(ValueError):
        top_k_neighbours(["a", "a"], np.eye(2), k=1, device="cpu")
    with pytest.raises(ValueError):
        top_k_neighbours(["a"], np.eye(2), k=1, device="cpu")


# -- retrieval --------------------------------------------------------------

def _paper(pid, connections=1):
    return {"id": pid, "title": "graph learning", "abstract": "graph learning", "graph_connections": connections}


def test_relative_gate_adapts_to_seed_similarity_band():
    papers = [_paper("near"), _paper("far")]
    scores = {"near": 0.83, "far": 0.70}
    kept = filter_symbolic_candidates("graph learning", papers, scores, mode="relative",
                                      seed_scores=[0.88, 0.85], margin=0.03)
    assert [paper["id"] for paper in kept] == ["near"]
    assert kept[0]["graph_filter_reason"] == "within_relative_margin"


def test_absolute_gate_rejects_neighbours_inside_neural_band():
    kept = filter_symbolic_candidates("graph learning", [_paper("near")], {"near": 0.83}, mode="absolute")
    assert kept == []


def test_relative_gate_without_seeds_retains_nothing():
    assert filter_symbolic_candidates("q", [_paper("x")], {"x": 0.99}, mode="relative", seed_scores=[]) == []


def test_expansion_pattern_allows_only_known_relationships():
    assert _expansion_pattern(["CITES", "SIMILAR_TO"]) == "CITES|SIMILAR_TO"
    with pytest.raises(ValueError):
        _expansion_pattern(["CITES]-() DETACH DELETE n //"])


def test_lexical_arm_changes_fusion_only_when_weighted():
    neural = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}]
    lexical = [{"id": "c", "score": 9.0}, {"id": "b", "score": 8.0}]
    baseline = fuse_results(neural, [], 3, lexical_papers=lexical, lexical_weight=0.0)
    assert [paper["id"] for paper in baseline] == ["a", "b"]
    fused = fuse_results(neural, [], 3, lexical_papers=lexical, lexical_weight=1.0)
    assert fused[0]["id"] == "b"  # supported by both arms
    assert {paper["id"] for paper in fused} == {"a", "b", "c"}
    assert next(p for p in fused if p["id"] == "c")["source"] == "lexical"


# -- embeddings -------------------------------------------------------------

def test_title_abstract_embedding_text_uses_original_case():
    row = {"title": "GNNs for Graphs", "abstract": "We Study X.", "clean_abstract": "we study x"}
    assert embedding_text(row, mode="abstract") == "we study x"
    assert embedding_text(row, mode="title_abstract", sep_token="[SEP]") == "GNNs for Graphs[SEP]We Study X."
    with pytest.raises(ValueError):
        embedding_text(row, mode="other")


# -- contradiction ----------------------------------------------------------

def test_structured_json_verdict_is_parsed():
    raw = json.dumps({"verdict": "contradiction", "confidence": 0.82, "reason": "Opposite effect.",
                      "claim_1": "A helps", "claim_2": "A hurts"})
    parsed = parse_contradiction_response(raw)
    assert parsed == {"verdict": "CONTRADICTION", "confidence": 0.82, "reason": "Opposite effect.", "valid": True}
    assert contradiction_verdict({"llm_analysis": raw}) == "CONTRADICTION"


def test_invalid_json_verdict_is_marked_invalid():
    parsed = parse_contradiction_response('{"verdict": "MAYBE", "confidence": 3}')
    assert parsed["valid"] is False and parsed["confidence"] is None


def test_legacy_line_format_still_parses():
    parsed = parse_contradiction_response("VERDICT: AGREEMENT\nCONFIDENCE: 0.6\nREASON: same result")
    assert parsed["verdict"] == "AGREEMENT" and parsed["valid"] is True


def test_year_weight_zero_ranks_by_concept_overlap_only():
    jaccard, gap, score = score_contradiction_candidate(2, 5, 5, 2020, 2025, year_weight=0.0)
    assert gap == 5 and score == pytest.approx(jaccard)
    _, _, legacy = score_contradiction_candidate(2, 5, 5, 2020, 2025, year_weight=0.25)
    assert legacy == pytest.approx(0.75 * jaccard + 0.25)
    with pytest.raises(ValueError):
        score_contradiction_candidate(2, 5, 5, 2020, 2025, year_weight=1.5)


# -- evaluation -------------------------------------------------------------

def test_condensed_ranking_drops_unjudged_documents():
    judgments = {"a": 2, "b": 0}
    assert condensed_ranking(["x", "a", "y", "b", "a"], judgments) == ["a", "b"]
    standard = evaluate_ranking(["x", "a"], judgments, (5,))
    condensed = evaluate_ranking(condensed_ranking(["x", "a"], judgments), judgments, (5,))
    assert condensed["ndcg@5"] > standard["ndcg@5"]


def test_variant_scoring_reports_unjudged_rate_and_both_views():
    benchmark = {"queries": [{"query_id": "Q1", "split": "dev", "query": "q",
                              "judgments": {"a": 2, "b": 1}}]}
    retrievers = {"v": lambda query, k: [{"id": "x"}, {"id": "a"}, {"id": "b"}]}
    summary, rankings = score_variants(benchmark, retrievers, split="dev", top_k=20)
    assert summary["v"]["unjudged_rate@10"] == pytest.approx(1 / 3)
    assert summary["v"]["condensed"]["ndcg@10"] == pytest.approx(1.0)
    assert summary["v"]["standard"]["ndcg@10"] < 1.0
    assert rankings[0]["ranked_paper_ids"] == ["x", "a", "b"]
    with pytest.raises(ValueError):
        score_variants(benchmark, retrievers, split="test")


def test_pool_extension_targets_only_unjudged_split_documents():
    from src.evaluation.pool_extension import merged_judgments, rescore, unjudged_rows

    benchmark = {"queries": [
        {"query_id": "D1", "split": "dev", "query": "q1", "judgments": {"a": 2}},
        {"query_id": "T1", "split": "test", "query": "q2", "judgments": {"z": 1}},
    ]}
    rankings = [{"variant": "v", "query_id": "D1", "ranked_paper_ids": ["a", "b"]},
                {"variant": "v", "query_id": "T1", "ranked_paper_ids": ["y"]}]
    corpus = {"b": {"clean_title": "B", "clean_abstract": "about b"}, "y": {"clean_title": "Y", "clean_abstract": "y"}}
    rows = unjudged_rows(benchmark, rankings, corpus, split="dev")
    assert [(row["query_id"], row["paper_id"], row["title"]) for row in rows] == [("D1", "b", "B")]

    merged = merged_judgments(benchmark, {"D1/b": {"grade": 2}})
    assert merged["D1"] == {"a": 2, "b": 2}
    summary = rescore(benchmark, rankings, merged, split="dev")
    assert summary["v"]["query_count"] == 1 and summary["v"]["unjudged_rate@20"] == 0
    with pytest.raises(ValueError):
        merged_judgments(benchmark, {"D1/a": {"grade": 0}})
    with pytest.raises(ValueError):
        unjudged_rows(benchmark, rankings, {}, split="dev")


def test_frozen_pool_digest_tolerates_only_line_ending_changes():
    original = b'{\n  "a": 1\n}\n'
    digest = hashlib.sha256(original).hexdigest()
    assert _matches_digest(original, digest)
    assert _matches_digest(original.replace(b"\n", b"\r\n"), digest)
    assert not _matches_digest(b'{\n  "a": 2\n}\n', digest)


def test_near_duplicate_collapse_keeps_highest_ranked_member():
    from src.pipeline.near_duplicates import are_near_duplicates, collapse

    mapping = {"a": frozenset({"b"}), "b": frozenset({"a"})}
    papers = [{"id": "b"}, {"id": "c"}, {"id": "a"}, {"id": "d"}]
    assert [p["id"] for p in collapse(papers, mapping)] == ["b", "c", "d"]
    assert are_near_duplicates("a", "b", mapping) and not are_near_duplicates("a", "c", mapping)


def test_near_duplicate_file_is_well_formed():
    from src.pipeline.near_duplicates import near_duplicate_map

    mapping = near_duplicate_map()
    assert mapping and all(key not in value for key, value in mapping.items())
