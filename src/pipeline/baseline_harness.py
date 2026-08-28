"""
src/pipeline/baseline_harness.py
================================
Baseline-comparison harness for NeSy-GraphRAG evaluation.

Runs the same set of queries through TWO retrieval paths:
  1. **NeSy (full)**   — ChromaDB neural retrieval + Neo4j symbolic expansion
  2. **Baseline**      — ChromaDB neural retrieval only (vector-only)

Then captures TS / NBR / ATD / RDI for each, diffs them, and prints a
comparison table.  This is the core quantitative result needed for the
dissertation's evaluation chapter.

Usage
-----
    python -m src.pipeline.baseline_harness

Or from code:
    from src.pipeline.baseline_harness import run_baseline_comparison
    results = run_baseline_comparison(queries)
"""

import sys
import os
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.pipeline.orchestrator import get_groq, get_neo4j
from src.pipeline.review import llm_review
from src.pipeline.metrics import compute_all_metrics
from src.pipeline.results_logger import log_result


# Default test queries (override by passing your own list)
DEFAULT_QUERIES = [
    "graph neural networks for node classification",
    "transformer architectures for natural language processing",
    "reinforcement learning in robotics",
    "knowledge graph embedding methods",
    "federated learning privacy preserving machine learning",
]


def run_baseline_comparison(
    queries: list[str] | None = None,
    top_k: int = 10,
) -> list[dict]:
    """Run every query through NeSy and baseline paths, collect metrics.

    Parameters
    ----------
    queries : list[str] — research queries to evaluate
    top_k   : int       — papers to retrieve per query

    Returns
    -------
    list[dict] — one entry per query with keys:
                 query, nesy_metrics, baseline_metrics, delta
    """
    queries = queries or DEFAULT_QUERIES
    driver  = get_neo4j()
    groq    = get_groq()

    results = []

    for i, query in enumerate(queries, 1):
        print(f"\n{'━'*70}")
        print(f"  QUERY {i}/{len(queries)}: {query}")
        print(f"{'━'*70}")

        # ── NeSy path ──
        print("\n  ▶ Running NeSy (full) path...")
        nesy_result = llm_review(groq, driver, query, top_k=top_k, baseline=False)
        nesy_metrics = compute_all_metrics(nesy_result, driver=driver)

        # ── Baseline path ──
        print("\n  ▶ Running Baseline (vector-only) path...")
        base_result = llm_review(groq, driver, query, top_k=top_k, baseline=True)
        base_metrics = compute_all_metrics(base_result, driver=driver)

        # ── Delta ──
        delta = {
            "ts"  : round(nesy_metrics["ts"]["ts"]   - base_metrics["ts"]["ts"],   4),
            "nbr" : round(nesy_metrics["nbr"]["nbr"] - base_metrics["nbr"]["nbr"], 4),
            "atd" : round(nesy_metrics["atd"]["atd"] - base_metrics["atd"]["atd"], 4),
            "rdi" : round(nesy_metrics["rdi"]["rdi"] - base_metrics["rdi"]["rdi"], 4),
        }

        entry = {
            "query"           : query,
            "nesy_metrics"    : nesy_metrics,
            "baseline_metrics": base_metrics,
            "nesy_sources"    : dict(Counter(p["source"] for p in nesy_result["papers"])),
            "baseline_sources": dict(Counter(p["source"] for p in base_result["papers"])),
            "delta"           : delta,
        }
        results.append(entry)
        print(f"  Sources: NeSy={entry['nesy_sources']}  Baseline={entry['baseline_sources']}")

        # Log both runs
        log_result(query, "nesy",     nesy_metrics)
        log_result(query, "baseline", base_metrics)

    # ── Print comparison table ──
    _print_comparison_table(results)
    return results


def _print_comparison_table(results: list[dict]) -> None:
    """Pretty-print the NeSy vs Baseline comparison table."""
    print("\n\n" + "═" * 90)
    print("  NeSy-GraphRAG vs Baseline — Evaluation Comparison")
    print("═" * 90)

    header = f"  {'Query':<45} {'Metric':>6}  {'NeSy':>7}  {'Base':>7}  {'Δ':>7}"
    print(header)
    print("  " + "─" * 86)

    for entry in results:
        q = entry["query"][:42] + "..." if len(entry["query"]) > 45 else entry["query"]
        n = entry["nesy_metrics"]
        b = entry["baseline_metrics"]
        d = entry["delta"]

        for metric_name, nesy_val, base_val, delta_val in [
            ("TS",  n["ts"]["ts"],   b["ts"]["ts"],   d["ts"]),
            ("NBR", n["nbr"]["nbr"], b["nbr"]["nbr"], d["nbr"]),
            ("ATD", n["atd"]["atd"], b["atd"]["atd"], d["atd"]),
            ("RDI", n["rdi"]["rdi"], b["rdi"]["rdi"], d["rdi"]),
        ]:
            sign = "+" if delta_val > 0 else ""
            marker = "✅" if delta_val > 0 else ("➖" if delta_val == 0 else "⚠️")
            label = q if metric_name == "TS" else ""
            print(f"  {label:<45} {metric_name:>6}  {nesy_val:>7.4f}  {base_val:>7.4f}  {sign}{delta_val:>6.4f} {marker}")

        print("  " + "─" * 86)

    # ── Averages ──
    n_queries = len(results)
    if n_queries > 0:
        avg_delta = {
            "ts"  : round(sum(r["delta"]["ts"]  for r in results) / n_queries, 4),
            "nbr" : round(sum(r["delta"]["nbr"] for r in results) / n_queries, 4),
            "atd" : round(sum(r["delta"]["atd"] for r in results) / n_queries, 4),
            "rdi" : round(sum(r["delta"]["rdi"] for r in results) / n_queries, 4),
        }
        print(f"\n  {'AVERAGE DELTA':<45}")
        for k, v in avg_delta.items():
            sign = "+" if v > 0 else ""
            print(f"  {'':45} {k.upper():>6}  {'':>7}  {'':>7}  {sign}{v:>6.4f}")

    print("═" * 90 + "\n")


# ── CLI entry point ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NeSy vs Baseline comparison harness")
    parser.add_argument("--top-k", type=int, default=10, help="Papers to retrieve per query")
    parser.add_argument("--queries", nargs="+", help="Custom queries (space-separated)")
    args = parser.parse_args()

    run_baseline_comparison(queries=args.queries, top_k=args.top_k)
