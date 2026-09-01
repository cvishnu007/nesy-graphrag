"""Offline-capable runner for Teammate 2 reasoning evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.claim_support_metrics import evaluate_claim_support
from src.evaluation.contradiction_runner import evaluate_contradictions
from src.evaluation.hypothesis_metrics import evaluate_hypotheses
from src.evaluation.reasoning_benchmark_io import (
    ReasoningBenchmarkValidationError,
    load_reasoning_benchmark,
    records_for_split,
)
from src.utils.config import (
    CLAIM_SUPPORT_BENCHMARK_FILE,
    CONTRADICTION_BENCHMARK_FILE,
    HYPOTHESIS_BENCHMARK_FILE,
    REASONING_RESULTS_DIR,
    SEMANTIC_SUPPORT_MIN_CONFIDENCE,
    SEMANTIC_SUPPORT_MODEL,
    is_configured,
)


TASKS = ("contradiction", "support", "hypothesis")


def _reference_metadata(benchmark: dict) -> dict:
    provenance = benchmark.get("annotation_provenance", {})
    return {
        "reference_annotation_source": provenance.get(
            "annotation_source", "unspecified"
        ),
        "human_ground_truth": provenance.get("human_ground_truth"),
        "independent_human_review": provenance.get("independent_human_review"),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def prerequisite_status() -> dict:
    """Describe live resources without connecting to or mutating them."""
    return {
        "frozen_corpus_present": Path("data/s2_clean.json").exists(),
        "chroma_directory_present": Path("data/chromadb").is_dir(),
        "neo4j_configured": all(
            is_configured(os.getenv(name), "YOUR_INSTANCE")
            for name in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
        ),
        "groq_configured": is_configured(os.getenv("GROQ_API_KEY")),
        "semantic_support_provider_configured": SEMANTIC_SUPPORT_MODEL != "unconfigured",
    }


def _read_jsonl(path: str | Path | None) -> list[dict] | None:
    if path is None:
        return None
    records = []
    try:
        with Path(path).open(encoding="utf-8") as file:
            for number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {number} must be a JSON object")
                records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read predictions {path}: {error}") from error
    return records


def _merge_predictions(gold: list[dict], predictions: list[dict], id_field: str) -> list[dict]:
    index = {}
    for prediction in predictions:
        identifier = prediction.get(id_field)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"Prediction requires non-empty {id_field}")
        if identifier in index:
            raise ValueError(f"Duplicate prediction {id_field}: {identifier}")
        index[identifier] = prediction
    gold_ids = {item[id_field] for item in gold}
    unknown = set(index).difference(gold_ids)
    if unknown:
        raise ValueError(f"Predictions reference unknown IDs: {sorted(unknown)}")
    return [{**item, **index.get(item[id_field], {})} for item in gold]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for value in values:
            file.write(json.dumps(value, ensure_ascii=False) + "\n")


def _write_hypothesis_csv(path: Path, hypotheses: list[dict]) -> None:
    fields = [
        "hypothesis_id", "query_id", "split", "reviewer_id", "evidence",
        "novelty", "feasibility", "specificity", "usefulness",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in hypotheses:
            for rating in item.get("ratings", []):
                writer.writerow({
                    "hypothesis_id": item["hypothesis_id"],
                    "query_id": item["query_id"],
                    "split": item["split"],
                    **{field: rating[field] for field in fields[3:]},
                })


def run_reasoning_evaluation(
    *, tasks: list[str], split: str, output_dir: str | Path,
    benchmark_paths: dict[str, str | Path] | None = None,
    prediction_paths: dict[str, str | Path | None] | None = None,
    overwrite: bool = False,
) -> dict:
    if not tasks or any(task not in TASKS for task in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError(f"tasks must be unique values from {TASKS}")
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    paths = benchmark_paths or {
        "contradiction": CONTRADICTION_BENCHMARK_FILE,
        "support": CLAIM_SUPPORT_BENCHMARK_FILE,
        "hypothesis": HYPOTHESIS_BENCHMARK_FILE,
    }
    predictions = prediction_paths or {}
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    failures = []
    metrics = {}
    versions = {}
    fixture_flags = {}
    reference_provenance = {}
    for task in tasks:
        try:
            benchmark = load_reasoning_benchmark(paths[task], task)
            versions[task] = benchmark["benchmark_version"]
            fixture_flags[task] = benchmark.get("fixture_only", False)
            reference_provenance[task] = _reference_metadata(benchmark)
            records = records_for_split(benchmark, task, split)
            if task == "hypothesis":
                _write_hypothesis_csv(directory / "hypothesis_ratings.csv", records)
                result = evaluate_hypotheses(records)
                _write_json(directory / "hypothesis_metrics.json", result)
                metrics[task] = result
                if not records:
                    failures.append({"task": task, "code": "no_benchmark_data", "message": f"No {split} hypothesis records"})
                continue

            prediction_rows = _read_jsonl(predictions.get(task))
            id_field = "pair_id" if task == "contradiction" else "item_id"
            merged = _merge_predictions(records, prediction_rows, id_field) if prediction_rows is not None else records
            raw_name = "contradiction_predictions.jsonl" if task == "contradiction" else "claim_support_predictions.jsonl"
            metric_name = "contradiction_metrics.json" if task == "contradiction" else "claim_support_metrics.json"
            _write_jsonl(directory / raw_name, merged)
            if not records:
                result = {"status": "no_benchmark_data", "count": 0}
                failures.append({"task": task, "code": "no_benchmark_data", "message": f"No {split} benchmark records"})
            elif prediction_rows is None:
                result = {"status": "predictions_unavailable", "count": len(records)}
                failures.append({"task": task, "code": "predictions_unavailable", "message": "No offline prediction file supplied; no live call was attempted"})
            else:
                result = evaluate_contradictions(merged) if task == "contradiction" else evaluate_claim_support(merged)
                result["status"] = "complete"
            _write_json(directory / metric_name, result)
            metrics[task] = result
        except (ReasoningBenchmarkValidationError, ValueError, KeyError) as error:
            failures.append({"task": task, "code": "evaluation_failure", "message": str(error)})
            metrics[task] = {"status": "failed", "error": str(error)}

    _write_jsonl(directory / "failures.jsonl", failures)
    metadata = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "tasks": tasks,
        "split": split,
        "benchmark_versions": versions,
        "fixture_only_by_task": fixture_flags,
        "reference_provenance_by_task": reference_provenance,
        "test_fixture_only": bool(fixture_flags) and all(fixture_flags.values()),
        "semantic_support_model": SEMANTIC_SUPPORT_MODEL,
        "semantic_support_min_confidence": SEMANTIC_SUPPORT_MIN_CONFIDENCE,
        "prerequisites": prerequisite_status(),
        "failure_count": len(failures),
        "offline_only": True,
    }
    _write_json(directory / "metadata.json", metadata)
    return {"metadata": metadata, "metrics": metrics, "failures": failures}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--output-dir", default=REASONING_RESULTS_DIR)
    parser.add_argument("--contradiction-benchmark", default=CONTRADICTION_BENCHMARK_FILE)
    parser.add_argument("--support-benchmark", default=CLAIM_SUPPORT_BENCHMARK_FILE)
    parser.add_argument("--hypothesis-benchmark", default=HYPOTHESIS_BENCHMARK_FILE)
    parser.add_argument("--contradiction-predictions")
    parser.add_argument("--support-predictions")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_reasoning_evaluation(
        tasks=arguments.tasks,
        split=arguments.split,
        output_dir=arguments.output_dir,
        benchmark_paths={
            "contradiction": arguments.contradiction_benchmark,
            "support": arguments.support_benchmark,
            "hypothesis": arguments.hypothesis_benchmark,
        },
        prediction_paths={
            "contradiction": arguments.contradiction_predictions,
            "support": arguments.support_predictions,
        },
        overwrite=arguments.overwrite,
    )
    print(json.dumps({"failure_count": len(result["failures"]), "output_dir": arguments.output_dir}, indent=2))
    return 0 if all(item["code"] in {"no_benchmark_data", "predictions_unavailable"} for item in result["failures"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
