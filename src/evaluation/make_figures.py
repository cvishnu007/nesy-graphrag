"""Paper figures, drawn from committed result files only (no stores, no models).

    python -m src.evaluation.make_figures

Writes to ``docs/figures/``:

* ``fig_topical_ndcg.{png,pdf,csv}``: NDCG@10 on the 44 stratified queries with
  bootstrap 95% CIs, for text retrieval, graph expansion, a cross-encoder reranker and
  the LightRAG-style baseline. Scored with the extended pool labels
  (``stratified_v2_lightrag_20261119``), a superset of the original pool, so that every
  system shown is fully judged at depth 20 (the selected system is 0.642 here and 0.644
  on the original pool).
* ``fig_graph_effect.{png,pdf,csv}``: the paired gain from adding the citation graph,
  per task, in each task's own metric, with bootstrap 95% CIs.
* ``architecture.{svg,png,pdf}`` is rendered from ``docs/figures/architecture.dot`` with
  Graphviz (``dot``) if it is installed.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

FIG = Path("docs/figures")
EXTENDED = Path("results/retrieval/stratified_v2_lightrag_20261119")
FROZEN = Path("results/retrieval/stratified_v2_20261114")
RERANK = Path("results/retrieval/reranker_20261120/rankings.jsonl")
QUERIES = Path("evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json")

# Reference palette (dataviz skill, light mode), first three categorical slots: validated all-pairs.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TEXT_ONLY, GRAPH, RERANKED = "#2a78d6", "#eb6834", "#1baf7a"


def _style(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none"})


def _bootstrap_ci(values, samples=10_000, seed=20261120):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def topical_rows():
    from src.evaluation.regrade_pool import score
    from src.evaluation.stratified_benchmark import load_rankings

    queries = json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
    ids = sorted(q["query_id"] for q in queries)
    labels = defaultdict(dict)
    for row in map(json.loads, (EXTENDED / "labels.jsonl").read_text().splitlines()):
        query_id, paper_id = row["item_id"].split("/", 1)
        labels[query_id][paper_id] = row["grade"]
    rankings = load_rankings(EXTENDED)
    if RERANK.exists():
        for row in map(json.loads, RERANK.read_text().splitlines()):
            rankings[row["variant"]][row["query_id"]] = row["ranked_paper_ids"]
    shown = [  # (variant, label, group)
        ("full_bge_bm25_similar", "BGE + BM25 + graph (selected)", "graph"),
        ("rerank_ms_marco_minilm_l_6_v2__full_bge_bm25_similar", "BGE + BM25 + graph, reranked", "reranked"),
        ("rerank_ms_marco_minilm_l_6_v2__rrf_bge_bm25", "BGE + BM25, reranked", "reranked"),
        ("bm25", "BM25", "text"),
        ("rrf_bge_bm25", "BGE + BM25", "text"),
        ("bge", "BGE", "text"),
        ("lightrag_mix", "LightRAG-style entity graph (mix)", "graph"),
        ("production_absolute", "Original system (SPECTER)", "text"),
    ]
    shown = [s for s in shown if s[0] in rankings]
    summary = score({v: rankings[v] for v, _, _ in shown}, labels, ids)
    rows = []
    for variant, label, group in shown:
        per_query = [summary[variant]["per_query_ndcg@10"][q] for q in ids]
        low, high = _bootstrap_ci(per_query)
        rows.append({"variant": variant, "label": label, "group": group, "ndcg@10": summary[variant]["ndcg@10"],
                     "ci_low": low, "ci_high": high, "queries": len(ids)})
    return sorted(rows, key=lambda r: r["ndcg@10"])


def graph_effect_rows():
    def load(path):
        return json.loads(Path(path).read_text())

    strat = load(FROZEN / "summary.json")["splits"]["all"]["comparisons"]["full_bge_bm25_similar_vs_rrf_bge_bm25"]
    dense = load("results/retrieval/dense_graph_20261115/summary_test.json")["comparisons"]["dense_citation_only_vs_rrf_bge_bm25"]
    citrec = load("results/retrieval/citation_recommendation_20261115/test_summary.json")["comparisons"][
        "rrf_bge_bm25_graph_k50_w1.0_vs_rrf_bge_bm25:recall@20"]
    temporal = load("results/reasoning/temporal_hypothesis_20261115/summary.json")["splits"]["all"]["comparisons"][
        "fusion_vector_graph_vs_vector:ap"]

    def cite(stem, enc):
        comparisons = load(f"results/retrieval/citeomatic_20261119/{stem}.json")["comparisons"]
        return comparisons[f"rrf_{enc}_bm25_cocitation_vs_rrf_{enc}_bm25_r@20"]

    entries = [  # top to bottom
        ("Topical retrieval, 44 q.", "NDCG@10", strat, "monte_carlo_two_sided_p"),
        ("Topical, densified graph, 22 q.", "NDCG@10", dense, "p"),
        ("Citation rec., our corpus, 400", "R@20", citrec, "p"),
        ("Citation rec., DBLP, BGE, 931", "R@20", cite("dblp_test", "bge"), "monte_carlo_two_sided_p"),
        ("DBLP, SPECTER2, 931", "R@20", cite("dblp_test_specter2_base", "specter2_base"), "monte_carlo_two_sided_p"),
        ("PubMed, BGE, 8,815", "R@20", cite("pubmed_test", "bge"), "monte_carlo_two_sided_p"),
        ("PubMed, SPECTER2, 8,815", "R@20", cite("pubmed_test_specter2_base", "specter2_base"), "monte_carlo_two_sided_p"),
        ("Time-sliced discovery, 120 tasks", "AP", temporal, "p"),
    ]
    return [{"task": task, "metric": metric, "delta": float(c["mean_delta"]),
             "ci_low": float(c["bootstrap_95_ci"][0]), "ci_high": float(c["bootstrap_95_ci"][1]), "p": float(c[p])}
            for task, metric, c, p in entries]


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)


def draw_topical(plt, rows):
    colors = {"text": TEXT_ONLY, "graph": GRAPH, "reranked": RERANKED}
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    y = np.arange(len(rows))
    values = [r["ndcg@10"] for r in rows]
    errors = np.array([[r["ndcg@10"] - r["ci_low"] for r in rows], [r["ci_high"] - r["ndcg@10"] for r in rows]])
    ax.barh(y, values, height=0.62, color=[colors[r["group"]] for r in rows], zorder=2)
    ax.errorbar(values, y, xerr=errors, fmt="none", ecolor=INK_2, elinewidth=1, capsize=2, zorder=3)
    for i, r in enumerate(rows):
        ax.text(r["ci_high"] + 0.012, i, f"{r['ndcg@10']:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("NDCG@10 on 44 topic-stratified queries (whiskers: bootstrap 95% CI)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (TEXT_ONLY, GRAPH, RERANKED)]
    fig.legend(handles, ["Text retrieval", "Uses a graph", "Cross-encoder reranked (MiniLM, top 20)"],
               loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def draw_graph_effect(plt, rows):
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color=INK_2, linewidth=1, zorder=1)
    for yi, r in zip(y, rows):
        excludes_zero = r["ci_low"] > 0 or r["ci_high"] < 0
        ax.plot([r["ci_low"], r["ci_high"]], [yi, yi], color=TEXT_ONLY, linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot(r["delta"], yi, "o", markersize=6.5, zorder=3, markeredgecolor=TEXT_ONLY, markeredgewidth=1.6,
                markerfacecolor=TEXT_ONLY if excludes_zero else SURFACE)
        ax.text(1.03, yi, f"{r['delta']:+.3f}  [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", va="center",
                fontsize=7.5, color=INK_2, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_yticks(y, [f"{r['task']} ({r['metric']})" for r in rows])
    ax.set_xlim(-0.06, 0.1)
    ax.set_xticks([-0.05, 0, 0.05, 0.1])
    ax.set_xlabel("Gain from the citation graph, in the task's metric\n(filled: 95% CI excludes 0)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.text(1.03, len(rows) - 0.35, "gain  [95% CI]", fontsize=7.5, color=INK_2, fontweight="bold",
            transform=ax.get_yaxis_transform(), clip_on=False)
    fig.subplots_adjust(left=0.37, right=0.74, top=0.9, bottom=0.2)
    return fig


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    FIG.mkdir(parents=True, exist_ok=True)
    for name, rows, draw in (("fig_topical_ndcg", topical_rows(), draw_topical),
                             ("fig_graph_effect", graph_effect_rows(), draw_graph_effect)):
        _write_csv(FIG / f"{name}.csv", rows)
        fig = draw(plt, rows)
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        print(f"wrote {FIG / name}.{{png,pdf,csv}}")
    dot = FIG / "architecture.dot"
    if dot.exists() and shutil.which("dot"):
        for ext in ("svg", "png", "pdf"):
            subprocess.run(["dot", f"-T{ext}", *(["-Gdpi=300"] if ext == "png" else []), str(dot),
                            "-o", str(FIG / f"architecture.{ext}")], check=True)
        print(f"wrote {FIG}/architecture.{{svg,png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
