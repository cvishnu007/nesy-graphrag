"""
run_test.py — Before/After comparison test runner
==================================================
Runs the full NeSy-GraphRAG pipeline and saves results to fixtures/.
Designed to work with both the old (master) and new (phase3) code.

Usage:
    python run_test.py before    # saves to fixtures/before/
    python run_test.py after     # saves to fixtures/after/
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline.orchestrator import graphrag_query, get_neo4j, get_groq
from src.pipeline.metrics import compute_all_metrics


TEST_QUERY = "graph neural networks for node classification"


def run_and_save(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'█'*60}")
    print(f"  RUNNING PIPELINE — saving to {output_dir}/")
    print(f"  Query: {TEST_QUERY}")
    print(f"{'█'*60}")

    # ── TEST 1: Literature Review ──
    print(f"\n{'='*60}")
    print("TEST 1 — LITERATURE REVIEW")
    print(f"{'='*60}")
    r1 = graphrag_query(TEST_QUERY, mode="review")

    # ── TEST 2: Contradiction Detection ──
    print(f"\n{'='*60}")
    print("TEST 2 — CONTRADICTION DETECTION")
    print(f"{'='*60}")
    r2 = graphrag_query(TEST_QUERY, mode="contradict")

    # ── TEST 3: Hypothesis Generation ──
    print(f"\n{'='*60}")
    print("TEST 3 — HYPOTHESIS GENERATION")
    print(f"{'='*60}")
    r3 = graphrag_query(TEST_QUERY, mode="hypothesis")

    # ── Compute combined metrics ──
    scores = compute_all_metrics(r1, contradiction_result=r2)

    # ── Save fixtures ──
    json.dump(r1,     open(os.path.join(output_dir, "review_result.json"), "w"),
              indent=2, default=str)
    json.dump(r2,     open(os.path.join(output_dir, "contradiction_result.json"), "w"),
              indent=2, default=str)
    json.dump(scores, open(os.path.join(output_dir, "metrics.json"), "w"),
              indent=2, default=str)

    print(f"\n\n{'═'*60}")
    print(f"  SAVED TO: {output_dir}/")
    print(f"  - review_result.json")
    print(f"  - contradiction_result.json")
    print(f"  - metrics.json")
    print(f"{'═'*60}")

    # ── Quick summary ──
    print(f"\n  METRICS SUMMARY:")
    print(f"  TS  = {scores['ts']['ts']}")
    print(f"  NBR = {scores['nbr']['nbr']}  (graph_count={scores['nbr']['graph_count']}, total={scores['nbr']['total']})")
    print(f"  ATD = {scores['atd']['atd']}  (years={scores['atd']['distinct_years']})")
    print(f"  RDI = {scores['rdi']['rdi']}  (contradictions_resolved={scores['rdi']['contradictions_resolved']})")

    # Source distribution
    sources = {}
    for p in r1.get("papers", []):
        s = p.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    print(f"\n  SOURCE DISTRIBUTION: {sources}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("before", "after"):
        print("Usage: python run_test.py before|after")
        sys.exit(1)

    label = sys.argv[1]
    output_dir = os.path.join("fixtures", label)
    run_and_save(output_dir)
