"""Holm correction for the nine predeclared comparisons on the 44-query benchmark,
with p-values precise enough to decide the borderline case.

The stratified benchmark (RESULTS §2d) used 20,000 sign-flip permutations. Selected
vs BGE alone came out at p = 0.0095 against a Holm threshold of 0.010, which is within
Monte Carlo noise of the line. This module recomputes all nine p-values with
1,000,000 permutations under two seeds from the committed rankings and judge labels,
applies Holm's step-down adjustment, and saves the result:

    python -m src.evaluation.holm_recheck

Output: ``results/retrieval/holm_recheck_20261120/summary.json``.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

BASE = Path("results/retrieval/stratified_v2_20261114")
QUERIES = Path("evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json")
OUT = Path("results/retrieval/holm_recheck_20261120")
SAMPLES = 1_000_000
SEEDS = (20261113, 20261120)
ALPHA = 0.05


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    ordered = sorted(p_values, key=lambda name: p_values[name])
    adjusted, running = {}, 0.0
    for i, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * p_values[name]))
        adjusted[name] = running
    return adjusted


def per_query_ndcg():
    """Per-query NDCG@10 of every stratified variant, recomputed from rankings and labels."""
    from src.evaluation.regrade_pool import score
    from src.evaluation.stratified_benchmark import load_rankings

    queries = json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
    labels = defaultdict(dict)
    for row in map(json.loads, (BASE / "labels.jsonl").read_text().splitlines()):
        query_id, paper_id = row["item_id"].split("/", 1)
        labels[query_id][paper_id] = row["grade"]
    ids = sorted(q["query_id"] for q in queries)
    summary = score(load_rankings(BASE), labels, ids)
    return ids, {name: data["per_query_ndcg@10"] for name, data in summary.items()}


def recheck(samples: int = SAMPLES, seeds=SEEDS) -> dict:
    from src.evaluation.significance import paired_permutation_test
    from src.evaluation.stratified_benchmark import COMPARISONS

    ids, ndcg = per_query_ndcg()
    by_seed = {}
    for seed in seeds:
        p = {}
        for challenger, reference in COMPARISONS:
            deltas = [ndcg[challenger][q] - ndcg[reference][q] for q in ids]
            p[f"{challenger}_vs_{reference}"] = paired_permutation_test(
                deltas, samples=samples, bootstrap_samples=1000, seed=seed)["monte_carlo_two_sided_p"]
        by_seed[str(seed)] = {"p": p, "holm_adjusted_p": holm(p)}
    names = list(by_seed[str(seeds[0])]["p"])
    return {
        "source": str(BASE), "queries": len(ids), "permutations": samples, "seeds": list(seeds), "alpha": ALPHA,
        "comparisons": {name: {
            "mean_delta": round(sum(ndcg[name.split("_vs_")[0]][q] - ndcg[name.split("_vs_")[1]][q]
                                    for q in ids) / len(ids), 4),
            "p": {seed: by_seed[seed]["p"][name] for seed in by_seed},
            "holm_adjusted_p": {seed: by_seed[seed]["holm_adjusted_p"][name] for seed in by_seed},
            "significant_after_holm": all(by_seed[seed]["holm_adjusted_p"][name] < ALPHA for seed in by_seed),
        } for name in names},
    }


def main() -> int:
    report = recheck()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, data in report["comparisons"].items():
        ps = " / ".join(f"{v:.5f}" for v in data["p"].values())
        adj = " / ".join(f"{v:.4f}" for v in data["holm_adjusted_p"].values())
        print(f"  {name:50} delta={data['mean_delta']:+.4f} p={ps} holm={adj} "
              f"{'significant' if data['significant_after_holm'] else 'n.s.'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
