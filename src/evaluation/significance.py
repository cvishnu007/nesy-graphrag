"""Paired uncertainty estimates for retrieval evaluation differences."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean


def paired_significance(
    deltas: list[float],
    *,
    bootstrap_samples: int = 50_000,
    seed: int = 20260830,
) -> dict:
    """Return paired bootstrap CI and an exact two-sided sign test."""
    if not deltas:
        raise ValueError("At least one paired difference is required")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    observed = fmean(deltas)
    rng = random.Random(seed)
    bootstrap = sorted(
        fmean(rng.choice(deltas) for _ in deltas)
        for _ in range(bootstrap_samples)
    )
    low = bootstrap[int(0.025 * bootstrap_samples)]
    high = bootstrap[int(0.975 * bootstrap_samples)]
    probability_positive = sum(value > 0 for value in bootstrap) / bootstrap_samples

    if len(deltas) > 20:
        raise ValueError("Exact randomization is limited to 20 pairs")
    extreme = 0
    assignment_count = 1 << len(deltas)
    for mask in range(assignment_count):
        randomized = fmean(
            delta if mask & (1 << index) else -delta
            for index, delta in enumerate(deltas)
        )
        if abs(randomized) >= abs(observed) - 1e-15:
            extreme += 1

    return {
        "pair_count": len(deltas),
        "mean_delta": observed,
        "bootstrap_95_ci": [low, high],
        "bootstrap_probability_positive": probability_positive,
        "exact_two_sided_randomization_p": extreme / assignment_count,
    }


def load_paired_deltas(path, challenger, reference, metric):
    """Read per-query metrics and return challenger-reference differences."""
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    grouped = defaultdict(dict)
    for row in rows:
        if metric not in row:
            raise ValueError(f"Metric column not found: {metric}")
        grouped[row["query_id"]][row["method"]] = float(row[metric])
    deltas = []
    for query_id, methods in sorted(grouped.items()):
        if challenger not in methods or reference not in methods:
            raise ValueError(f"Missing method for query {query_id}")
        deltas.append(methods[challenger] - methods[reference])
    return deltas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv")
    parser.add_argument("--challenger", default="hybrid")
    parser.add_argument("--reference", default="bm25")
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    deltas = load_paired_deltas(
        arguments.metrics_csv,
        arguments.challenger,
        arguments.reference,
        arguments.metric,
    )
    result = {
        "challenger": arguments.challenger,
        "reference": arguments.reference,
        "metric": arguments.metric,
        **paired_significance(deltas),
    }
    text = json.dumps(result, indent=2) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
