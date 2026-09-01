"""Checkpointed, blinded AI evaluation of claim/passage semantic support.

Reference labels and notes are never included in model prompts. Outputs are
explicitly AI-generated predictions evaluated against AI-reference annotations.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.evaluation.claim_support_metrics import evaluate_claim_support
from src.evaluation.reasoning_benchmark_io import (
    load_reasoning_benchmark,
    records_for_split,
)
from src.evaluation.semantic_support import nli_scores_to_prediction, parse_support_decision
from src.utils.config import GROQ_API_KEY
from src.utils.groq_client import groq_chat_with_retry


class SupportAIEvaluationError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_support_prompt(items: list[Mapping[str, Any]]) -> str:
    visible = [{
        "item_id": item["item_id"],
        "claim": item["claim"],
        "passage_id": item["passage_id"],
        "passage_text": item["passage_text"],
        "paper_id": item["paper_id"],
    } for item in items]
    return f"""Produce AI-generated semantic-support predictions, not human judgments.
Use only each visible claim and passage. Choose exactly one label:
SUPPORTED (complete claim directly entailed), PARTIALLY_SUPPORTED (material but incomplete support),
UNSUPPORTED (related or silent), or CONTRADICTED (conflicts with the claim).
Confidence must be a number from 0 to 1. Return strict JSON only:
{{"annotations":[{{"id":"ITEM_ID","label":"SUPPORTED","confidence":0.9,"reason":"brief evidence-based reason"}}]}}
Return exactly one annotation for every input ID and no extras.
VISIBLE RECORDS:
{json.dumps(visible, ensure_ascii=False)}"""


def parse_support_batch(raw: Any, expected_ids: list[str], *, model: str) -> list[dict]:
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as error:
            raise SupportAIEvaluationError(f"invalid JSON: {error}") from error
    annotations = raw.get("annotations") if isinstance(raw, Mapping) else None
    if not isinstance(annotations, list):
        raise SupportAIEvaluationError("output must contain an annotations list")
    parsed = {}
    for item in annotations:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise SupportAIEvaluationError("every annotation requires a string id")
        item_id = item["id"]
        if item_id in parsed:
            raise SupportAIEvaluationError(f"duplicate annotation ID: {item_id}")
        decision = parse_support_decision(item, passage_id="benchmark-passage", model=model)
        if not decision["valid"]:
            raise SupportAIEvaluationError(f"malformed decision for {item_id}")
        parsed[item_id] = {
            "item_id": item_id,
            "prediction": decision["label"],
            "confidence": decision["confidence"],
            "valid": True,
            "reason": decision["reason"],
            "model": decision["model"],
            "annotation_source": "AI-generated prediction",
            "reference_source": "AI-generated reference annotation",
            "human_review": False,
            "created_at_utc": _now(),
        }
    if set(parsed) != set(expected_ids):
        raise SupportAIEvaluationError(
            f"prediction IDs differ: expected={sorted(expected_ids)}, got={sorted(parsed)}"
        )
    return [parsed[item_id] for item_id in expected_ids]


def generate_local_nli_predictions(
    items: list[dict],
    output_path: str | Path,
    *,
    model_name: str = "cross-encoder/nli-deberta-v3-small",
    batch_size: int = 16,
    partial_entailment_floor: float = 0.20,
) -> list[dict]:
    """Run a cached pretrained NLI model without network access."""
    import torch
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        model_name,
        local_files_only=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    probabilities = model.predict(
        [(item["passage_text"], item["claim"]) for item in items],
        batch_size=batch_size,
        show_progress_bar=True,
        apply_softmax=True,
    )
    id2label = {
        int(index): label.casefold() for index, label in model.model.config.id2label.items()
    }
    results = []
    for item, values in zip(items, probabilities):
        scores = {id2label[index]: float(value) for index, value in enumerate(values)}
        mapped = nli_scores_to_prediction(
            scores, partial_entailment_floor=partial_entailment_floor
        )
        results.append({
            "item_id": item["item_id"],
            "prediction": mapped["label"],
            "confidence": mapped["confidence"],
            "valid": mapped["valid"],
            "nli_scores": scores,
            "reason": "Deterministic mapping from local NLI class probabilities.",
            "model": model_name,
            "annotation_source": "AI model-generated prediction (local NLI)",
            "reference_source": "AI-generated reference annotation",
            "human_review": False,
            "created_at_utc": _now(),
        })
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(path, results)
    return results


def _write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _read_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def generate_predictions(
    items: list[dict],
    output_path: str | Path,
    judge: Callable[[str], Any],
    *,
    model: str,
    batch_size: int = 6,
    max_attempts: int = 6,
) -> list[dict]:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = _read_checkpoint(path)
    existing = {row.get("item_id") for row in results}
    expected = {item["item_id"] for item in items}
    unknown = existing.difference(expected)
    if unknown:
        raise SupportAIEvaluationError(f"checkpoint contains unknown IDs: {sorted(unknown)}")
    pending = [item for item in items if item["item_id"] not in existing]
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        prompt = build_support_prompt(batch)
        error = None
        for _ in range(max_attempts):
            try:
                generated = parse_support_batch(
                    judge(prompt), [item["item_id"] for item in batch], model=model
                )
                break
            except SupportAIEvaluationError as exc:
                error = exc
                prompt += f"\nCORRECTION REQUIRED: {exc}. Return complete strict JSON again."
        else:
            raise SupportAIEvaluationError(f"could not parse batch: {error}")
        results.extend(generated)
        _write_jsonl(path, results)
    order = {item["item_id"]: index for index, item in enumerate(items)}
    return sorted(results, key=lambda row: order[row["item_id"]])


def select_confidence_threshold(rows: list[dict], *, thresholds=None) -> dict:
    candidates = []
    for threshold in thresholds or [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
        evaluated = [
            {**row, "valid": bool(row.get("valid")) and row.get("confidence", 0) >= threshold}
            for row in rows
        ]
        metrics = evaluate_claim_support(evaluated)
        candidates.append({"threshold": threshold, "metrics": metrics})
    best = max(
        candidates,
        key=lambda item: (
            item["metrics"]["macro_f1"], item["metrics"]["coverage"],
            -item["threshold"],
        ),
    )
    return {
        "selection_split": "dev",
        "reference_source": "AI-generated reference annotations",
        "human_review": False,
        "selected_threshold": best["threshold"],
        "candidates": candidates,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="evaluation/benchmarks/claim_support.json")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen/qwen3.6-27b")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--provider", choices=("local-nli", "groq"), default="local-nli")
    parser.add_argument("--partial-entailment-floor", type=float, default=0.20)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--threshold-report")
    args = parser.parse_args(argv)
    benchmark = load_reasoning_benchmark(args.benchmark, "support")
    items = records_for_split(benchmark, "support", args.split)
    if args.provider == "local-nli":
        model = (
            args.model if args.model != "qwen/qwen3.6-27b"
            else "cross-encoder/nli-deberta-v3-small"
        )
        rows = generate_local_nli_predictions(
            items, args.output, model_name=model, batch_size=args.batch_size,
            partial_entailment_floor=args.partial_entailment_floor,
        )
    else:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        judge = lambda prompt: groq_chat_with_retry(
            client, prompt, model=args.model, max_tokens=2000,
            temperature=0.0, max_retries=6,
        )
        rows = generate_predictions(
            items, args.output, judge, model=args.model, batch_size=args.batch_size
        )
    if args.split == "dev" and args.threshold_report:
        gold = {item["item_id"]: item["label"] for item in items}
        report = select_confidence_threshold([
            {**row, "label": gold[row["item_id"]]} for row in rows
        ])
        report_path = Path(args.threshold_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        args.confidence_threshold = report["selected_threshold"]
    if args.confidence_threshold is not None:
        rows = [
            {
                **row,
                "valid": bool(row.get("valid"))
                and row.get("confidence", 0) >= args.confidence_threshold,
                "frozen_confidence_threshold": args.confidence_threshold,
            }
            for row in rows
        ]
        _write_jsonl(Path(args.output), rows)
    print(json.dumps({"split": args.split, "predictions": len(rows), "output": args.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
