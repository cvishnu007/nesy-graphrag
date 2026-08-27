"""
diff_results.py — Compare before/after fixture metrics
=======================================================
Reads fixtures/before/metrics.json and fixtures/after/metrics.json,
prints a side-by-side comparison table.

Usage:
    python diff_results.py
"""
import json
import os


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    before_dir = os.path.join("fixtures", "before")
    after_dir  = os.path.join("fixtures", "after")

    # ── Load metrics ──
    before_metrics = load_json(os.path.join(before_dir, "metrics.json"))
    after_metrics  = load_json(os.path.join(after_dir, "metrics.json"))

    # ── Load review results for source analysis ──
    before_review = load_json(os.path.join(before_dir, "review_result.json"))
    after_review  = load_json(os.path.join(after_dir, "review_result.json"))

    # ── Load contradiction results ──
    before_contra = load_json(os.path.join(before_dir, "contradiction_result.json"))
    after_contra  = load_json(os.path.join(after_dir, "contradiction_result.json"))

    # ═══════════════════════════════════════════
    # METRICS COMPARISON
    # ═══════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  NeSy-GraphRAG — BEFORE / AFTER COMPARISON")
    print("═" * 70)

    print(f"\n  {'Metric':<35} {'BEFORE':>10} {'AFTER':>10} {'DELTA':>10}")
    print("  " + "─" * 65)

    for metric in ["ts", "nbr", "atd", "rdi"]:
        b_val = before_metrics[metric][metric]
        a_val = after_metrics[metric][metric]
        delta = round(a_val - b_val, 4)
        sign = "+" if delta > 0 else ""
        marker = " ✅" if delta != 0 else ""
        print(f"  {metric.upper():<35} {b_val:>10.4f} {a_val:>10.4f} {sign}{delta:>9.4f}{marker}")

    # ── NBR detail ──
    print(f"\n  {'NBR Detail':<35} {'BEFORE':>10} {'AFTER':>10}")
    print("  " + "─" * 55)
    for key in ["graph_count", "neural_only_count", "total"]:
        b = before_metrics["nbr"][key]
        a = after_metrics["nbr"][key]
        print(f"  {key:<35} {b:>10} {a:>10}")

    # ── RDI detail ──
    print(f"\n  {'RDI Detail':<35} {'BEFORE':>10} {'AFTER':>10}")
    print("  " + "─" * 55)
    for key in ["cross_doc_papers", "contradictions_resolved", "total_checked"]:
        b = before_metrics["rdi"][key]
        a = after_metrics["rdi"][key]
        print(f"  {key:<35} {b:>10} {a:>10}")

    # ═══════════════════════════════════════════
    # SOURCE DISTRIBUTION
    # ═══════════════════════════════════════════
    print(f"\n  {'Source Distribution':<35} {'BEFORE':>10} {'AFTER':>10}")
    print("  " + "─" * 55)

    def count_sources(papers):
        dist = {"neural": 0, "symbolic": 0, "both": 0}
        for p in papers:
            s = p.get("source", "unknown")
            if s in dist:
                dist[s] += 1
            else:
                dist[s] = dist.get(s, 0) + 1
        return dist

    b_dist = count_sources(before_review.get("papers", []))
    a_dist = count_sources(after_review.get("papers", []))

    for src in ["neural", "symbolic", "both"]:
        print(f"  {src:<35} {b_dist.get(src, 0):>10} {a_dist.get(src, 0):>10}")

    # ═══════════════════════════════════════════
    # CONTRADICTION ANALYSIS
    # ═══════════════════════════════════════════
    print(f"\n  Contradiction Verdicts:")
    print("  " + "─" * 55)

    for label, data in [("BEFORE", before_contra), ("AFTER", after_contra)]:
        contradictions = data.get("contradictions", [])
        print(f"\n  [{label}] {len(contradictions)} pairs checked:")
        for i, item in enumerate(contradictions):
            analysis = item.get("llm_analysis", "")
            # Extract verdict line
            verdict = "UNKNOWN"
            for line in analysis.split("\n"):
                if "VERDICT:" in line.upper():
                    verdict = line.strip()
                    break
            p1 = item.get("paper1", {}).get("title", "?")[:40]
            p2 = item.get("paper2", {}).get("title", "?")[:40]
            print(f"    Pair {i+1}: {verdict}")

    # ═══════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  KEY FINDINGS:")
    print("═" * 70)

    nbr_before = before_metrics["nbr"]["nbr"]
    nbr_after  = after_metrics["nbr"]["nbr"]
    print(f"  NBR: {'FIXED ✅' if nbr_after != nbr_before else 'UNCHANGED ⚠️'}"
          f" — was {nbr_before:.4f}, now {nbr_after:.4f}")

    rdi_before = before_metrics["rdi"]["rdi"]
    rdi_after  = after_metrics["rdi"]["rdi"]
    print(f"  RDI: {'FIXED ✅' if rdi_after != rdi_before else 'UNCHANGED ⚠️'}"
          f" — was {rdi_before:.4f}, now {rdi_after:.4f}")

    b_both = b_dist.get("both", 0)
    a_both = a_dist.get("both", 0)
    print(f"  Source 'both': was {b_both}, now {a_both}"
          f" — {'FIXED ✅' if a_both != b_both else 'SAME'}")

    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
