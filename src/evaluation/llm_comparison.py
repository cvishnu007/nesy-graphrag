"""Controlled, checkpointed comparison of two LLM contradiction judges."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping

from src.evaluation.contradiction_runner import PRIMARY_CLASSES, evaluate_contradictions
from src.evaluation.reasoning_benchmark_io import load_reasoning_benchmark, records_for_split
from src.utils.groq_client import groq_chat_with_retry


DEFAULT_MODELS = {
    "qwen_27b": "qwen/qwen3.6-27b",
    "qwen_8b": "qwen/qwen3.8-27b",
}
MAX_PARSE_ATTEMPTS = 4


def build_llm_comparison_prompt(records: list[dict]) -> str:
    """Build a blinded prompt containing evidence but no reference annotations."""
    visible = [{
        "id": row["pair_id"],
        "paper1_id": row["paper1_id"],
        "paper1_title": row.get("paper1_title", ""),
        "paper1_abstract": row.get("paper1_abstract", ""),
        "paper2_id": row["paper2_id"],
        "paper2_title": row.get("paper2_title", ""),
        "paper2_abstract": row.get("paper2_abstract", ""),
    } for row in records]
    return (
        "You are an AI reference judge comparing scientific papers. Classify each pair as "
        "CONTRADICTION, AGREEMENT, or DIFFERENT SCOPE. A contradiction requires incompatible "
        "claims under comparable conditions; topic overlap alone is insufficient. Return only "
        "valid JSON with this shape: {\"annotations\":[{\"id\":\"...\","
        "\"prediction\":\"AGREEMENT\",\"confidence\":0.0,\"reason\":\"brief evidence-based reason\"}]}. "
        "Confidence must be from 0 to 1 and every supplied id must occur exactly once.\n\n"
        + json.dumps(visible, ensure_ascii=False, separators=(",", ":"))
    )


def _parse_response(raw: str, expected_ids: list[str]) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    data = json.loads(text)
    annotations = data.get("annotations") if isinstance(data, dict) else None
    if not isinstance(annotations, list):
        raise ValueError("response must contain an annotations list")
    by_id = {}
    for item in annotations:
        if not isinstance(item, dict) or item.get("id") in by_id:
            raise ValueError("annotations must be unique objects")
        item_id = item.get("id")
        prediction = item.get("prediction")
        confidence = item.get("confidence")
        reason = item.get("reason")
        if item_id not in expected_ids or prediction not in PRIMARY_CLASSES:
            raise ValueError("invalid id or prediction")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be non-empty")
        by_id[item_id] = {
            "pair_id": item_id,
            "prediction": prediction,
            "confidence": float(confidence),
            "reason": reason.strip(),
            "valid": True,
        }
    if set(by_id) != set(expected_ids):
        raise ValueError("response ids do not exactly match requested ids")
    return [by_id[item_id] for item_id in expected_ids]


def _load_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["pair_id"]] = row
    return rows


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def run_llm_comparison(
    records: list[dict],
    judges: Mapping[str, Callable[[str], str]],
    model_ids: Mapping[str, str],
    *,
    output_dir: str | Path,
    split: str,
    batch_size: int = 4,
    threshold: float = 0.5,
) -> dict:
    """Run or resume identical blinded prompts against each model."""
    if set(judges) != set(model_ids) or len(judges) != 2:
        raise ValueError("exactly two matching judges and model IDs are required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ids = [row.get("pair_id") for row in records]
    if len(set(ids)) != len(ids) or any(row.get("split") != split for row in records):
        raise ValueError("records must have unique IDs and match the requested split")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    failures_path = output / "failures.jsonl"
    batches = [records[index:index + batch_size] for index in range(0, len(records), batch_size)]
    prompts = [build_llm_comparison_prompt(batch) for batch in batches]
    prompt_hashes = [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts]
    models = {}
    for alias, judge in judges.items():
        checkpoint = output / f"{alias}_predictions.jsonl"
        saved = _load_jsonl(checkpoint)
        initial_count = len(saved)
        calls = 0
        runtime = 0.0
        invalid_responses = 0
        for batch, prompt in zip(batches, prompts):
            batch_ids = [row["pair_id"] for row in batch]
            if all(item_id in saved for item_id in batch_ids):
                continue
            parsed = None
            last_error = None
            for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
                started = perf_counter()
                raw = judge(prompt)
                runtime += perf_counter() - started
                calls += 1
                try:
                    parsed = _parse_response(raw, batch_ids)
                    break
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    last_error = error
                    invalid_responses += 1
                    failure = {
                        "model_alias": alias,
                        "model_id": model_ids[alias],
                        "batch_ids": batch_ids,
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "raw_response_sha256": hashlib.sha256(str(raw).encode("utf-8")).hexdigest(),
                        "raw_response_length": len(raw) if isinstance(raw, str) else None,
                    }
                    with failures_path.open("a", encoding="utf-8") as file:
                        file.write(json.dumps(failure, ensure_ascii=False) + "\n")
            if parsed is None:
                raise ValueError(
                    f"model {alias} returned invalid output after {MAX_PARSE_ATTEMPTS} attempts: {last_error}"
                )
            for prediction in parsed:
                prediction.update({
                    "model_alias": alias,
                    "model_id": model_ids[alias],
                    "annotation_source": "AI-generated model prediction",
                    "human_ground_truth": False,
                })
                saved[prediction["pair_id"]] = prediction
            _write_jsonl_atomic(checkpoint, [saved[item_id] for item_id in ids if item_id in saved])
        if set(saved) != set(ids):
            raise ValueError(f"checkpoint {checkpoint} does not exactly cover the benchmark")
        evaluated = [{**row, **saved[row["pair_id"]]} for row in records]
        models[alias] = {
            "model_id": model_ids[alias],
            "prompt_hashes": prompt_hashes,
            "metrics": evaluate_contradictions(evaluated, threshold=threshold),
            "prediction_count": len(saved),
            "preserved_prediction_count": initial_count,
            "new_api_call_count": calls,
            "invalid_response_count": invalid_responses,
            "new_runtime_seconds": runtime,
        }
    result = {
        "split": split,
        "record_count": len(records),
        "threshold": threshold,
        "models": models,
        "reference_annotation_source": "AI-generated reference annotations",
        "human_ground_truth": False,
        "controlled_variables": {
            "only_changed_component": "llm_model",
            "identical_records": True,
            "identical_blinded_prompts": True,
            "temperature": 0.0,
            "max_tokens": 1000,
            "reasoning_effort": "none",
            "max_parse_attempts": MAX_PARSE_ATTEMPTS,
            "batch_size": batch_size,
        },
    }
    (output / "comparison_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures_path.touch(exist_ok=True)
    return result


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _joined_records(benchmark_path: str, pool_path: str, split: str) -> list[dict]:
    benchmark = load_reasoning_benchmark(benchmark_path, "contradiction")
    references = {row["pair_id"]: row for row in records_for_split(benchmark, "contradiction", split)}
    pool_data = json.loads(Path(pool_path).read_text(encoding="utf-8"))
    pool_rows = {row["pair_id"]: row for row in pool_data["records"] if row["split"] == split}
    if set(references) != set(pool_rows):
        raise ValueError("frozen benchmark and blinded pool IDs differ")
    return [{
        **{key: value for key, value in pool_rows[item_id].items() if key not in {"annotations", "adjudication"}},
        "label": reference["label"],
    } for item_id, reference in references.items()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-a", default=DEFAULT_MODELS["qwen_27b"])
    parser.add_argument("--model-b", default=DEFAULT_MODELS["qwen_8b"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    from groq import Groq
    from src.utils.config import GROQ_API_KEY

    records = _joined_records(args.benchmark, args.pool, args.split)
    client = Groq(api_key=GROQ_API_KEY)
    models = {"model_a": args.model_a, "model_b": args.model_b}
    judges = {
        alias: (lambda prompt, model=model: groq_chat_with_retry(
            client, prompt, model=model, max_tokens=1000, temperature=0.0,
            max_retries=6, reasoning_effort="none",
        ))
        for alias, model in models.items()
    }
    result = run_llm_comparison(records, judges, models, output_dir=args.output_dir,
                                split=args.split, batch_size=args.batch_size, threshold=args.threshold)
    benchmark_sha = hashlib.sha256(Path(args.benchmark).read_bytes()).hexdigest()
    pool_sha = hashlib.sha256(Path(args.pool).read_bytes()).hexdigest()
    metadata = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "benchmark": args.benchmark,
        "benchmark_sha256": benchmark_sha,
        "blinded_pool": args.pool,
        "blinded_pool_sha256": pool_sha,
        "split": args.split,
        "models": models,
        "human_ground_truth": False,
        "reference_annotation_source": "AI-generated reference annotations",
        "controlled_variables": result["controlled_variables"],
        "reproduction_command": " ".join([".\\venv\\Scripts\\python.exe", "-m", "src.evaluation.llm_comparison", *(__import__('sys').argv[1:])]),
    }
    output = Path(args.output_dir)
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({alias: data["metrics"] for alias, data in result["models"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
