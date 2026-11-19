"""Recompute headline numbers from saved artifacts and check them against docs/RESULTS.md.

No LLM calls: judge labels, rankings, and human sheets are read from ``results/``
and ``evaluation/human_review/``. Exits non-zero if any value differs from the
expected one by more than 0.0015. Run from the repository root:

    python -m src.evaluation.reproduce
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

TOLERANCE = 0.0015


def judge_kappa():
    from src.evaluation.human_agreement import pair_report
    from src.evaluation.judge_agreement import split_items

    rows = list(csv.DictReader(open("evaluation/human_review/relevance_sheet_cv.csv", encoding="utf-8-sig")))
    human = {r["item_id"]: int(r["grade_0_1_2"]) for r in rows}
    _, held_out = split_items(list(human))
    out = {}
    for name, path in (("7b_strict_batch1", "results/retrieval/judge_agreement_local_20261114/held_out_ollama_qwen2_5_7b_instruct_strict_b1.jsonl"),
                       ("27b_strict", "results/retrieval/judge_agreement_groq_20261114/held_out_groq_qwen_qwen3_8_27b_strict.jsonl")):
        grades = {r["item_id"]: r["grade"] for r in map(json.loads, Path(path).read_text().splitlines())}
        out[name] = pair_report([human[i] for i in sorted(held_out)], [grades[i] for i in sorted(held_out)])["quadratic_weighted_kappa"]
    return out


def human_human_kappa():
    from src.evaluation.human_agreement import pair_report

    key = json.loads(Path("evaluation/human_review/relevance_subset50_key_DO_NOT_SHARE_WITH_ANNOTATORS.json").read_text())
    mapping = {k["item_id"]: k["source_item_id"] for k in key["items"]}
    first = {r["item_id"]: int(r["grade_0_1_2"]) for r in csv.DictReader(open("evaluation/human_review/relevance_sheet_cv.csv", encoding="utf-8-sig"))}
    second = {mapping[r["item_id"]]: int(r["grade_0_1_2"]) for r in csv.DictReader(open("evaluation/human_review/relevance_subset50_sheet_rater3.csv", encoding="utf-8-sig"))}
    items = sorted(second)
    return pair_report([first[i] for i in items], [second[i] for i in items])["quadratic_weighted_kappa"]


def stratified():
    from src.evaluation.regrade_pool import score
    from src.evaluation.stratified_benchmark import load_rankings

    base = Path("results/retrieval/stratified_v2_20261114")
    queries = json.loads(Path("evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json").read_text(encoding="utf-8"))["queries"]
    graded = {r["item_id"]: r["grade"] for r in map(json.loads, (base / "labels.jsonl").read_text().splitlines())}
    labels = defaultdict(dict)
    for item_id, grade in graded.items():
        q, p = item_id.split("/", 1)
        labels[q][p] = grade
    summary = score(load_rankings(base), labels, sorted(q["query_id"] for q in queries))
    return {name: summary[name]["ndcg@10"] for name in ("full_bge_bm25_similar", "bm25", "rrf_bge_bm25", "bge", "production_absolute")}


def regraded_gnn():
    report = json.loads(Path("results/retrieval/regraded_qwen2_5_7b_strict_b1_20261114/summary.json").read_text())
    variants = report["splits"]["test"]["variants"]
    return {name: variants[name]["ndcg@10"] for name in ("full_bge_bm25_similar", "bge", "production_absolute")}


def saved(path, *keys):
    value = json.loads(Path(path).read_text())
    for key in keys:
        value = value[key]
    return value


EXPECTED = {
    "judge held-out kappa, 7B strict 1/call": 0.7485,
    "judge held-out kappa, 27B strict": 0.8837,
    "human-human kappa, 50-pair subset": 0.9084,
    "hypothesis round 1 relevance kappa, 2 raters": 0.9047,
    "contradiction round 1 kappa, 2 raters": 0.7601,
    "hypothesis round 2 relevance, bridge minus vector": -0.5385,
    "contradiction round 2, verdict-flagged tension+ precision": 0.556,
    "LightRAG-style mix NDCG@10 (44 queries)": 0.396,
    "DBLP test F1@20, BGE+BM25+co-citation": 0.284,
    "PubMed test F1@20, BGE+BM25+co-citation": 0.305,
    "stratified NDCG@10, selected": 0.644,
    "stratified NDCG@10, BM25": 0.626,
    "stratified NDCG@10, BGE+BM25": 0.621,
    "stratified NDCG@10, BGE": 0.593,
    "stratified NDCG@10, original": 0.287,
    "GNN test NDCG@10, selected (regraded)": 0.813,
    "GNN test NDCG@10, original (regraded)": 0.349,
    "citation rec. test Recall@20, with graph": 0.308,
    "citation rec. test Recall@20, BGE+BM25": 0.241,
    "time-sliced AP, vector+graph": 0.0825,
    "time-sliced AP, vector": 0.0436,
    "time-sliced P@10, vector+graph": 0.0667,
}


def main() -> int:
    kappa = judge_kappa()
    strat = stratified()
    gnn = regraded_gnn()
    cit = "results/retrieval/citation_recommendation_20261115/test_summary.json"
    tsl = "results/reasoning/temporal_hypothesis_20261115/summary.json"
    actual = {
        "judge held-out kappa, 7B strict 1/call": kappa["7b_strict_batch1"],
        "judge held-out kappa, 27B strict": kappa["27b_strict"],
        "human-human kappa, 50-pair subset": human_human_kappa(),
        "hypothesis round 1 relevance kappa, 2 raters": saved(
            "evaluation/human_review/hypothesis_report_cv_rater3.json", "inter_rater_kappa",
            "hypothesis_sheet_rated_cv~hypothesis_sheet_rated_rater3", "relevance_1_5"),
        "contradiction round 1 kappa, 2 raters": saved(
            "evaluation/human_review/contradiction_report_cv_rater3.json", "inter_rater_kappa",
            "contradiction_sheet_labeled_cv~contradiction_sheet_labeled_rater3", "label_0_1_2"),
        "hypothesis round 2 relevance, bridge minus vector": saved(
            "evaluation/human_review/hypothesis_round2_report_cv.json", "comparisons",
            "bridge_m10_minus_vector_neighbour", "relevance_1_5", "mean_delta"),
        "contradiction round 2, verdict-flagged tension+ precision": saved(
            "evaluation/human_review/contradiction_round2_report_cv.json", "arms",
            "index_verdict_contradiction", "precision_contradiction_or_tension"),
        "LightRAG-style mix NDCG@10 (44 queries)": saved(
            "results/retrieval/stratified_v2_lightrag_20261119/summary.json", "splits", "all", "variants",
            "lightrag_mix", "ndcg@10"),
        "DBLP test F1@20, BGE+BM25+co-citation": saved(
            "results/retrieval/citeomatic_20261119/dblp_test.json", "methods", "rrf_bge_bm25_cocitation", "f1@20"),
        "PubMed test F1@20, BGE+BM25+co-citation": saved(
            "results/retrieval/citeomatic_20261119/pubmed_test.json", "methods", "rrf_bge_bm25_cocitation", "f1@20"),
        "stratified NDCG@10, selected": strat["full_bge_bm25_similar"],
        "stratified NDCG@10, BM25": strat["bm25"],
        "stratified NDCG@10, BGE+BM25": strat["rrf_bge_bm25"],
        "stratified NDCG@10, BGE": strat["bge"],
        "stratified NDCG@10, original": strat["production_absolute"],
        "GNN test NDCG@10, selected (regraded)": gnn["full_bge_bm25_similar"],
        "GNN test NDCG@10, original (regraded)": gnn["production_absolute"],
        "citation rec. test Recall@20, with graph": saved(cit, "variants", "rrf_bge_bm25_graph_k50_w1.0", "recall@20"),
        "citation rec. test Recall@20, BGE+BM25": saved(cit, "variants", "rrf_bge_bm25", "recall@20"),
        "time-sliced AP, vector+graph": saved(tsl, "splits", "all", "scorers", "fusion_vector_graph", "ap"),
        "time-sliced AP, vector": saved(tsl, "splits", "all", "scorers", "vector", "ap"),
        "time-sliced P@10, vector+graph": saved(tsl, "splits", "all", "scorers", "fusion_vector_graph", "p@10"),
    }
    failures = 0
    for name, expected in EXPECTED.items():
        ok = abs(actual[name] - expected) <= TOLERANCE
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {name:45} expected {expected:.4f}  got {actual[name]:.4f}")
    print(f"\n{len(EXPECTED) - failures}/{len(EXPECTED)} reproduced")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
