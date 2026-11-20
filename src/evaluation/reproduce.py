"""Recompute headline numbers from saved artifacts and check them against docs/RESULTS.md.

No LLM calls and no databases: judge labels, rankings and human sheets are read from
``results/`` and ``evaluation/human_review/``. Each check says how it got its value:

* ``recomputed``: from raw committed files (rankings + labels, rating sheets + keys
  rebuilt from fixed seeds, per-query metrics), with the project's own metric code;
* ``saved``: read from a saved summary, because the per-query results behind it were
  not kept (the citation-recommendation, DBLP/PubMed and time-sliced runs predate
  per-query output) or the key cannot be rebuilt (hypothesis round 2). These checks
  catch transcription errors only. They switch to ``recomputed`` automatically once
  the per-query files exist (rerun the benchmark from the local stores) or the key is
  present.

Exits non-zero if any value differs from the expected one by more than 0.0015.
Run from the repository root:

    python -m src.evaluation.reproduce
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path
from statistics import fmean

TOLERANCE = 0.0015
REVIEW = Path("evaluation/human_review")
CITREC = Path("results/retrieval/citation_recommendation_20261115")
CITEOMATIC = Path("results/retrieval/citeomatic_20261119")
TEMPORAL = Path("results/reasoning/temporal_hypothesis_20261115")


def _module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sheet(name):
    return list(csv.DictReader(open(REVIEW / name, encoding="utf-8-sig")))


def saved(path, *keys):
    value = json.loads(Path(path).read_text())
    for key in keys:
        value = value[key]
    return value


def fisher_two_sided(a, n1, b, n2):
    """Two-sided Fisher exact p for a/n1 vs b/n2 (sum of tables no more likely than the observed one)."""
    total, hits = n1 + n2, a + b

    def prob(x):
        return comb(n1, x) * comb(n2, hits - x) / comb(total, hits)

    observed = prob(a)
    return min(1.0, sum(prob(x) for x in range(max(0, hits - n2), min(hits, n1) + 1)
                        if prob(x) <= observed * (1 + 1e-7)))


# --- relevance judge and human agreement ---------------------------------------------

def judge_kappa():
    from src.evaluation.human_agreement import pair_report
    from src.evaluation.judge_agreement import split_items

    human = {r["item_id"]: int(r["grade_0_1_2"]) for r in _sheet("relevance_sheet_cv.csv")}
    _, held_out = split_items(list(human))
    out = {}
    for name, path in (("7b_strict_batch1", "results/retrieval/judge_agreement_local_20261114/held_out_ollama_qwen2_5_7b_instruct_strict_b1.jsonl"),
                       ("27b_strict", "results/retrieval/judge_agreement_groq_20261114/held_out_groq_qwen_qwen3_8_27b_strict.jsonl")):
        grades = {r["item_id"]: r["grade"] for r in map(json.loads, Path(path).read_text().splitlines())}
        out[name] = pair_report([human[i] for i in sorted(held_out)], [grades[i] for i in sorted(held_out)])["quadratic_weighted_kappa"]
    return out


def frozen_3b_kappa():
    """The frozen 3B pool labels against the first annotator, on all 200 pairs (RESULTS §2b)."""
    from src.evaluation.human_agreement import pair_report

    _, key = _module(REVIEW / "build_relevance_sheet.py").build()  # deterministic; writes nothing
    ai = {k["item_id"]: k["ai_grade"] for k in key}
    human = {r["item_id"]: int(r["grade_0_1_2"]) for r in _sheet("relevance_sheet_cv.csv")}
    items = sorted(human)
    return pair_report([human[i] for i in items], [ai[i] for i in items])["quadratic_weighted_kappa"]


def human_human_kappa():
    from src.evaluation.human_agreement import pair_report

    # The S -> R key is gitignored (annotator blinding) but deterministic; rebuild it.
    mapping = _module(REVIEW / "build_relevance_subset.py").subset_mapping()
    first = {r["item_id"]: int(r["grade_0_1_2"]) for r in _sheet("relevance_sheet_cv.csv")}
    second = {mapping[r["item_id"]]: int(r["grade_0_1_2"]) for r in _sheet("relevance_subset50_sheet_rater3.csv")}
    items = sorted(second)
    return pair_report([first[i] for i in items], [second[i] for i in items])["quadratic_weighted_kappa"]


def judge_transfer_kappa():
    from src.evaluation.human_agreement import pair_report

    builder = _module(REVIEW / "build_judge_transfer_sheet.py")
    mapping, labels = builder.subset_mapping(), builder.judge_labels()
    human = {mapping[r["item_id"]]: int(r["grade_0_1_2"]) for r in _sheet("judge_transfer_sheet_raterC.csv")}
    items = sorted(human)
    kappa = pair_report([human[i] for i in items], [labels[i] for i in items])["quadratic_weighted_kappa"]
    by_t = {t: human[item] for t, item in mapping.items()}
    large = builder.extra_judges()["groq_qwen_qwen3_8_27b_strict"]
    ts = sorted(by_t)
    return kappa, pair_report([by_t[t] for t in ts], [large[t] for t in ts])["quadratic_weighted_kappa"]


# --- human ratings of hypotheses and contradictions ----------------------------------

def human_ratings():
    """Arm-level human results, re-scored from the rating sheets with keys rebuilt from the kit seeds."""
    from src.evaluation import human_ratings as hr

    keys = _module(REVIEW / "rebuild_arm_keys.py").rebuilt_keys()
    fields, scale = hr.HYPOTHESIS_FIELDS, {"1", "2", "3", "4", "5"}
    hyp = {n: hr.read(REVIEW / f"{n}.csv", fields, scale)
           for n in ("hypothesis_sheet_rated_cv", "hypothesis_sheet_rated_rater3")}
    con = {n: hr.read(REVIEW / f"{n}.csv", ("label_0_1_2",), {"0", "1", "2"})
           for n in ("contradiction_sheet_labeled_cv", "contradiction_sheet_labeled_rater3")}
    con_a = {"contradiction_sheet_labeled_cv": con["contradiction_sheet_labeled_cv"]}
    round2 = {"contradiction_round2_sheet_labeled_cv": hr.read(
        REVIEW / "contradiction_round2_sheet_labeled_cv.csv", ("label_0_1_2",), {"0", "1", "2"})}
    h1 = hr.hypothesis_report(hyp, keys["hypothesis"])
    c1_both = hr.contradiction_report(con, keys["contradiction"])
    c1 = hr.contradiction_report(con_a, keys["contradiction"])
    c2 = hr.contradiction_report(round2, keys["contradiction_round2"])

    def count(report, arm, field="precision_contradiction_or_tension"):
        return round(report["arms"][arm][field] * report["arms"][arm]["n"]), report["arms"][arm]["n"]

    flagged, top_unflagged, lower = (count(c2, a) for a in
                                     ("index_verdict_contradiction", "index_top150_unflagged", "index_rank151_plus"))
    flagged_contra = count(c2, "index_verdict_contradiction", "precision_contradiction")
    screen = count(c1, "top_ranked")
    screen_contra = count(c1, "top_ranked", "precision_contradiction")
    pair = "hypothesis_sheet_rated_cv~hypothesis_sheet_rated_rater3"
    return {
        "hyp_kappa": h1["inter_rater_kappa"][pair]["relevance_1_5"],
        "con_kappa": c1_both["inter_rater_kappa"]["contradiction_sheet_labeled_cv~contradiction_sheet_labeled_rater3"]["label_0_1_2"],
        "hyp_graph": h1["arms"]["graph_structural_hole"]["relevance_1_5"],
        "hyp_vector": h1["arms"]["vector_neighbour"]["relevance_1_5"],
        "hyp_p": h1["graph_minus_vector"]["relevance_1_5"]["monte_carlo_two_sided_p"],
        "screen": screen[0] / screen[1],
        "flagged": flagged[0] / flagged[1],
        "fisher_flagged_vs_screen": fisher_two_sided(*flagged, *screen),
        "fisher_contra_vs_screen": fisher_two_sided(*flagged_contra, *screen_contra),
        "fisher_top150_vs_lower": fisher_two_sided(flagged[0] + top_unflagged[0], flagged[1] + top_unflagged[1], *lower),
        "fisher_flagged_vs_unflagged": fisher_two_sided(*flagged, *top_unflagged),
    }


def hypothesis_round2():
    """Bridge minus vector relevance. Recomputed only if the (non-rebuildable) key is present."""
    key_path = REVIEW / "hypothesis_round2_key_DO_NOT_SHARE_WITH_ANNOTATORS.json"
    if not key_path.exists():
        return saved(REVIEW / "hypothesis_round2_report_cv.json", "comparisons",
                     "bridge_m10_minus_vector_neighbour", "relevance_1_5", "mean_delta"), "saved"
    from src.evaluation import human_ratings as hr

    sheets = {"hypothesis_round2_sheet_rated_cv": hr.read(REVIEW / "hypothesis_round2_sheet_rated_cv.csv",
                                                          hr.HYPOTHESIS_FIELDS, {"1", "2", "3", "4", "5"})}
    report = hr.hypothesis_round2_report(sheets, json.loads(key_path.read_text()))
    return report["comparisons"]["bridge_m10_minus_vector_neighbour"]["relevance_1_5"]["mean_delta"], "recomputed"


# --- retrieval -----------------------------------------------------------------------

def _labels(path):
    labels = defaultdict(dict)
    for row in map(json.loads, Path(path).read_text().splitlines()):
        query_id, paper_id = row["item_id"].split("/", 1)
        labels[query_id][paper_id] = row["grade"]
    return labels


def stratified():
    from src.evaluation.regrade_pool import score
    from src.evaluation.stratified_benchmark import load_rankings

    base = Path("results/retrieval/stratified_v2_20261114")
    queries = json.loads(Path("evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json").read_text(encoding="utf-8"))["queries"]
    summary = score(load_rankings(base), _labels(base / "labels.jsonl"), sorted(q["query_id"] for q in queries))
    return {name: summary[name]["ndcg@10"] for name in ("full_bge_bm25_similar", "bm25", "rrf_bge_bm25", "bge", "production_absolute")}


def holm_selected_vs_bge():
    from src.evaluation.holm_recheck import SEEDS, recheck

    report = recheck(seeds=SEEDS[:1])
    return report["comparisons"]["full_bge_bm25_similar_vs_bge"]["holm_adjusted_p"][str(SEEDS[0])]


def lightrag_mix():
    from src.evaluation.regrade_pool import score
    from src.evaluation.stratified_benchmark import load_rankings

    base = Path("results/retrieval/stratified_v2_lightrag_20261119")
    queries = json.loads(Path("evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json").read_text(encoding="utf-8"))["queries"]
    rankings = load_rankings(base)
    summary = score({"lightrag_mix": rankings["lightrag_mix"]}, _labels(base / "labels.jsonl"),
                    sorted(q["query_id"] for q in queries))
    return summary["lightrag_mix"]["ndcg@10"]


def regraded_gnn():
    from src.evaluation.regrade_pool import BENCHMARK, load_rankings, score

    benchmark = json.loads(Path(BENCHMARK).read_text())
    test_ids = sorted(q["query_id"] for q in benchmark["queries"] if q["split"] == "test")
    labels = _labels("results/retrieval/regraded_qwen2_5_7b_strict_b1_20261114/labels.jsonl")
    summary = score(load_rankings("test"), labels, test_ids)
    return {name: summary[name]["ndcg@10"] for name in ("full_bge_bm25_similar", "production_absolute")}


def citation_recommendation(variant):
    """Test Recall@20 for a variant: from per-query metrics if saved, else the summary."""
    per_query = CITREC / "test_per_query.json"
    if per_query.exists():
        values = json.loads(per_query.read_text())[variant]
        return fmean(v["recall@20"] for v in values.values()), "recomputed"
    return saved(CITREC / "test_summary.json", "variants", variant, "recall@20"), "saved"


def citeomatic(stem, method):
    """F1@20 on a DBLP/PubMed run: from per-query metrics if saved, else the summary."""
    per_query = CITEOMATIC / f"{stem}_per_query.json"
    if per_query.exists():
        from src.evaluation.citeomatic_benchmark import summarise

        return summarise(json.loads(per_query.read_text())[method])["f1@20"], "recomputed"
    return saved(CITEOMATIC / f"{stem}.json", "methods", method, "f1@20"), "saved"


def temporal(scorer, metric):
    """Mean over all 120 tasks: from per-task metrics if saved, else the summary."""
    per_task = TEMPORAL / "per_task_summary.json"
    if per_task.exists():
        values = json.loads(per_task.read_text())["scorers"][scorer]
        return fmean(v[metric] for v in values.values()), "recomputed"
    return saved(TEMPORAL / "summary.json", "splits", "all", "scorers", scorer, metric), "saved"


# --- checks --------------------------------------------------------------------------

EXPECTED = {
    "judge held-out kappa, 7B strict 1/call": 0.7485,
    "judge held-out kappa, 27B strict": 0.8837,
    "frozen 3B pool labels kappa, all 200 pairs": 0.1867,
    "human-human kappa, 50-pair subset": 0.9084,
    "judge transfer kappa, 10 non-GNN topics": 0.5714,
    "judge transfer kappa, 27B, 10 non-GNN topics": 0.8759,
    "hypothesis round 1 relevance kappa, 2 raters": 0.9047,
    "contradiction round 1 kappa, 2 raters": 0.7601,
    "hypothesis round 1 relevance, graph arm (2 raters)": 3.100,
    "hypothesis round 1 relevance, vector arm (2 raters)": 4.117,
    "hypothesis round 1 relevance, graph vs vector p": 0.0010,
    "hypothesis round 2 relevance, bridge minus vector": -0.5385,
    "contradiction round 1, screened tension+ precision (A)": 0.0893,
    "contradiction round 2, verdict-flagged tension+ precision": 0.5556,
    "contradiction Fisher p, flagged 5/9 vs screened 5/56": 0.0029,
    "contradiction Fisher p, contradictions 2/9 vs 0/56": 0.0173,
    "contradiction Fisher p, top-150 10/24 vs rank 151+ 1/16": 0.0274,
    "contradiction Fisher p, flagged 5/9 vs unflagged 5/15": 0.4028,
    "LightRAG-style mix NDCG@10 (44 queries)": 0.396,
    "DBLP test F1@20, BGE+BM25+co-citation": 0.284,
    "PubMed test F1@20, BGE+BM25+co-citation": 0.305,
    "DBLP test F1@20, SPECTER2+BM25+co-citation": 0.300,
    "stratified NDCG@10, selected": 0.644,
    "stratified NDCG@10, BM25": 0.626,
    "stratified NDCG@10, BGE+BM25": 0.621,
    "stratified NDCG@10, BGE": 0.593,
    "stratified NDCG@10, original": 0.287,
    "stratified Holm-adjusted p, selected vs BGE (1M perm.)": 0.052,
    "GNN test NDCG@10, selected (regraded)": 0.813,
    "GNN test NDCG@10, original (regraded)": 0.349,
    "citation rec. test Recall@20, with graph": 0.308,
    "citation rec. test Recall@20, BGE+BM25": 0.241,
    "time-sliced AP, vector+graph": 0.0825,
    "time-sliced AP, vector": 0.0436,
    "time-sliced P@10, vector+graph": 0.0667,
}


def actual_values():
    """{check name: (value, 'recomputed' | 'saved')}."""
    kappa = judge_kappa()
    transfer = judge_transfer_kappa()
    ratings = human_ratings()
    strat = stratified()
    gnn = regraded_gnn()
    r = "recomputed"
    return {
        "judge held-out kappa, 7B strict 1/call": (kappa["7b_strict_batch1"], r),
        "judge held-out kappa, 27B strict": (kappa["27b_strict"], r),
        "frozen 3B pool labels kappa, all 200 pairs": (frozen_3b_kappa(), r),
        "human-human kappa, 50-pair subset": (human_human_kappa(), r),
        "judge transfer kappa, 10 non-GNN topics": (transfer[0], r),
        "judge transfer kappa, 27B, 10 non-GNN topics": (transfer[1], r),
        "hypothesis round 1 relevance kappa, 2 raters": (ratings["hyp_kappa"], r),
        "contradiction round 1 kappa, 2 raters": (ratings["con_kappa"], r),
        "hypothesis round 1 relevance, graph arm (2 raters)": (ratings["hyp_graph"], r),
        "hypothesis round 1 relevance, vector arm (2 raters)": (ratings["hyp_vector"], r),
        "hypothesis round 1 relevance, graph vs vector p": (ratings["hyp_p"], r),
        "hypothesis round 2 relevance, bridge minus vector": hypothesis_round2(),
        "contradiction round 1, screened tension+ precision (A)": (ratings["screen"], r),
        "contradiction round 2, verdict-flagged tension+ precision": (ratings["flagged"], r),
        "contradiction Fisher p, flagged 5/9 vs screened 5/56": (ratings["fisher_flagged_vs_screen"], r),
        "contradiction Fisher p, contradictions 2/9 vs 0/56": (ratings["fisher_contra_vs_screen"], r),
        "contradiction Fisher p, top-150 10/24 vs rank 151+ 1/16": (ratings["fisher_top150_vs_lower"], r),
        "contradiction Fisher p, flagged 5/9 vs unflagged 5/15": (ratings["fisher_flagged_vs_unflagged"], r),
        "LightRAG-style mix NDCG@10 (44 queries)": (lightrag_mix(), r),
        "DBLP test F1@20, BGE+BM25+co-citation": citeomatic("dblp_test", "rrf_bge_bm25_cocitation"),
        "PubMed test F1@20, BGE+BM25+co-citation": citeomatic("pubmed_test", "rrf_bge_bm25_cocitation"),
        "DBLP test F1@20, SPECTER2+BM25+co-citation": citeomatic("dblp_test_specter2_base",
                                                                 "rrf_specter2_base_bm25_cocitation"),
        "stratified NDCG@10, selected": (strat["full_bge_bm25_similar"], r),
        "stratified NDCG@10, BM25": (strat["bm25"], r),
        "stratified NDCG@10, BGE+BM25": (strat["rrf_bge_bm25"], r),
        "stratified NDCG@10, BGE": (strat["bge"], r),
        "stratified NDCG@10, original": (strat["production_absolute"], r),
        "stratified Holm-adjusted p, selected vs BGE (1M perm.)": (holm_selected_vs_bge(), r),
        "GNN test NDCG@10, selected (regraded)": (gnn["full_bge_bm25_similar"], r),
        "GNN test NDCG@10, original (regraded)": (gnn["production_absolute"], r),
        "citation rec. test Recall@20, with graph": citation_recommendation("rrf_bge_bm25_graph_k50_w1.0"),
        "citation rec. test Recall@20, BGE+BM25": citation_recommendation("rrf_bge_bm25"),
        "time-sliced AP, vector+graph": temporal("fusion_vector_graph", "ap"),
        "time-sliced AP, vector": temporal("vector", "ap"),
        "time-sliced P@10, vector+graph": temporal("fusion_vector_graph", "p@10"),
    }


def main() -> int:
    actual = actual_values()
    failures = 0
    for name, expected in EXPECTED.items():
        value, how = actual[name]
        ok = abs(value - expected) <= TOLERANCE
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {name:56} expected {expected:.4f}  got {value:.4f}  [{how}]")
    recomputed = sum(how == "recomputed" for _, how in actual.values())
    print(f"\n{len(EXPECTED) - failures}/{len(EXPECTED)} reproduced "
          f"({recomputed} recomputed from raw files, {len(EXPECTED) - recomputed} read from saved summaries)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
