"""Compare passage-existence and semantic support on frozen AI references."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.evaluation.claim_support_metrics import evaluate_claim_support
from src.evaluation.reasoning_benchmark_io import (
    load_reasoning_benchmark,
    records_for_split,
)


DELTA_METRICS = (
    "accuracy", "macro_precision", "macro_recall", "macro_f1", "coverage",
    "false_acceptance_rate", "unsupported_claim_rejection_rate",
)


def existence_only_predictions(items: Iterable[dict]) -> list[dict]:
    rows = []
    for item in items:
        if not isinstance(item.get("passage_id"), str) or not item["passage_id"].strip():
            raise ValueError("Existence-only baseline requires a passage ID")
        rows.append({
            "item_id": item["item_id"],
            "prediction": "SUPPORTED",
            "confidence": 1.0,
            "valid": True,
            "baseline": "passage_id_existence_only",
        })
    return rows


def _prediction_index(predictions: Iterable[dict]) -> dict[str, dict]:
    index = {}
    for row in predictions:
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("Every semantic prediction requires item_id")
        if item_id in index:
            raise ValueError(f"Duplicate semantic prediction ID: {item_id}")
        index[item_id] = dict(row)
    return index


def compare_support_baselines(
    items: Iterable[dict], semantic_predictions: Iterable[dict], *, threshold: float
) -> dict:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    records = list(items)
    expected = {item["item_id"] for item in records}
    semantic_index = _prediction_index(semantic_predictions)
    if set(semantic_index) != expected:
        raise ValueError(
            f"Semantic prediction IDs differ; missing={sorted(expected - set(semantic_index))}, "
            f"extra={sorted(set(semantic_index) - expected)}"
        )
    existence_index = {
        row["item_id"]: row for row in existence_only_predictions(records)
    }
    existence_rows = [{**item, **existence_index[item["item_id"]]} for item in records]
    semantic_rows = []
    for item in records:
        prediction = semantic_index[item["item_id"]]
        confidence = prediction.get("confidence", 0.0)
        valid = (
            bool(prediction.get("valid", True))
            and isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and confidence >= threshold
        )
        semantic_rows.append({**item, **prediction, "valid": valid})
    existence_metrics = evaluate_claim_support(existence_rows)
    semantic_metrics = evaluate_claim_support(semantic_rows)
    return {
        "threshold": float(threshold),
        "selection_split": "dev",
        "reference_annotation_source": "AI-generated",
        "human_ground_truth": False,
        "existence_only": existence_metrics,
        "semantic": semantic_metrics,
        "semantic_minus_existence": {
            name: semantic_metrics[name] - existence_metrics[name]
            for name in DELTA_METRICS
        },
        "existence_predictions": existence_rows,
        "semantic_predictions": semantic_rows,
    }


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--semantic-predictions", required=True)
    parser.add_argument("--threshold-report", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    benchmark = load_reasoning_benchmark(args.benchmark, "support")
    items = records_for_split(benchmark, "support", args.split)
    threshold_report = json.loads(Path(args.threshold_report).read_text(encoding="utf-8"))
    threshold = threshold_report.get("selected_threshold")
    result = compare_support_baselines(
        items, _read_jsonl(Path(args.semantic_predictions)), threshold=threshold,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {key: value for key, value in result.items() if not key.endswith("_predictions")}
    _write_json(output / "comparison_metrics.json", metrics)
    _write_jsonl(output / "existence_predictions.jsonl", result["existence_predictions"])
    _write_jsonl(output / "semantic_predictions.jsonl", result["semantic_predictions"])
    audited = [
        row for row in result["semantic_predictions"]
        if not row["valid"] or row.get("prediction") in {"UNSUPPORTED", "CONTRADICTED"}
    ]
    _write_jsonl(output / "semantic_rejection_audit.jsonl", audited)
    (output / "failures.jsonl").write_text("", encoding="utf-8")
    command = (
        ".\\venv\\Scripts\\python.exe -m src.evaluation.support_baseline_comparison "
        f"--benchmark {args.benchmark} --semantic-predictions {args.semantic_predictions} "
        f"--threshold-report {args.threshold_report} --split {args.split} "
        f"--output-dir {args.output_dir}"
    )
    _write_json(output / "metadata.json", {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "task": "support_baseline_comparison",
        "split": args.split,
        "benchmark_version": benchmark["benchmark_version"],
        "threshold": threshold,
        "threshold_selection_split": "dev",
        "reference_annotation_source": "AI-generated",
        "human_ground_truth": False,
        "failure_count": 0,
        "reproduction_command": command,
    })
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
