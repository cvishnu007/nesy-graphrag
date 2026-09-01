"""Evaluate contradiction candidate-stage recall on frozen AI references."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from src.evaluation.contradiction_runner import canonical_pair, candidate_recall_at_k
from src.evaluation.reasoning_benchmark_io import (
    load_reasoning_benchmark,
    records_for_split,
)


def _candidate_pair(record: dict) -> tuple[str, str]:
    try:
        return canonical_pair(record["paper1"]["id"], record["paper2"]["id"])
    except (KeyError, TypeError) as error:
        raise ValueError("Candidate requires paper1.id and paper2.id") from error


def evaluate_candidate_recall(
    references: Iterable[dict],
    candidates: Iterable[dict],
    k_values: Sequence[int] = (1, 5, 10, 20, 50, 100),
) -> dict:
    refs = list(references)
    rows = list(candidates)
    if not k_values or any(
        isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in k_values
    ):
        raise ValueError("k_values must contain positive integers")
    reference_pairs = []
    reference_ids = {}
    for reference in refs:
        pair = canonical_pair(reference.get("paper1_id"), reference.get("paper2_id"))
        if pair in reference_ids:
            raise ValueError(f"Duplicate reference unordered pair: {pair}")
        reference_pairs.append(pair)
        reference_ids[pair] = reference.get("pair_id")
    ranked_pairs = [_candidate_pair(row) for row in rows]
    if len(set(ranked_pairs)) != len(ranked_pairs):
        raise ValueError("Duplicate unordered pair in candidates")
    ranked_set = set(ranked_pairs)
    missing = sorted(
        reference_ids[pair] for pair in reference_pairs if pair not in ranked_set
    )
    result = {
        "reference_count": len(reference_pairs),
        "candidate_count": len(ranked_pairs),
        "recovered_count": len(reference_pairs) - len(missing),
        "missing_count": len(missing),
        "missing_reference_ids": missing,
        "full_pool_recall": (
            (len(reference_pairs) - len(missing)) / len(reference_pairs)
            if reference_pairs else 0.0
        ),
    }
    for k in k_values:
        result[f"recall@{k}"] = candidate_recall_at_k(
            reference_pairs, ranked_pairs, k
        )
    return result


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Every candidate JSONL row must be an object")
    return rows


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_evaluation(
    *, benchmark_path: Path, candidate_path: Path, split: str,
    output_dir: Path, command: str,
) -> dict:
    benchmark = load_reasoning_benchmark(benchmark_path, "contradiction")
    references = records_for_split(benchmark, "contradiction", split)
    candidates = [
        row for row in _read_jsonl(candidate_path) if row.get("split") == split
    ]
    metrics = evaluate_candidate_recall(references, candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "candidate_metrics.json", {
        **metrics,
        "split": split,
        "benchmark_version": benchmark["benchmark_version"],
        "reference_annotation_source": "AI-generated",
        "human_ground_truth": False,
    })
    reference_by_id = {row["pair_id"]: row for row in references}
    with (output_dir / "missing_pairs.jsonl").open("w", encoding="utf-8") as file:
        for pair_id in metrics["missing_reference_ids"]:
            file.write(json.dumps(reference_by_id[pair_id], ensure_ascii=False) + "\n")
    (output_dir / "failures.jsonl").write_text("", encoding="utf-8")
    _write_json(output_dir / "metadata.json", {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "task": "contradiction_candidate_recall",
        "split": split,
        "benchmark": str(benchmark_path),
        "benchmark_version": benchmark["benchmark_version"],
        "candidates": str(candidate_path),
        "reference_annotation_source": "AI-generated",
        "human_ground_truth": False,
        "failure_count": 0,
        "reproduction_command": command,
    })
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    command = (
        ".\\venv\\Scripts\\python.exe -m src.evaluation.contradiction_candidate_evaluator "
        f"--benchmark {args.benchmark} --candidates {args.candidates} "
        f"--split {args.split} --output-dir {args.output_dir}"
    )
    metrics = run_evaluation(
        benchmark_path=Path(args.benchmark), candidate_path=Path(args.candidates),
        split=args.split, output_dir=Path(args.output_dir), command=command,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
