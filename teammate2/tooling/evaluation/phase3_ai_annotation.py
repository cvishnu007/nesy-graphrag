"""Populate Phase 3 reviewer packets with explicitly AI-generated references.

This is a methodological fallback, not human annotation. The implementation reads
only blinded packet JSON and never reads Phase 2 protected system sidecars.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.annotation_pool import finalize_annotation_pool
from src.evaluation.phase3_annotation import (
    HYPOTHESIS_DIMENSIONS,
    ID_FIELDS,
    POOL_FILES,
    Phase3AnnotationError,
    _load_pools,
    _read_json,
    _response_key,
    _validate_response,
    _write_json,
    build_annotated_pools,
    load_completed_packets,
    validate_reviewer_packet,
)


AI_SOURCE = "AI-generated"
PREEXISTING_RESPONSE_IDS = ["C0C22508E24E7", "C0D27B651D266"]
DEFAULT_CHUNKS = {"contradiction": 4, "support": 6, "hypothesis": 8}
DEFAULT_MAX_PROMPT_CHARS = 8000
Judge = Callable[[str], Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rubric(task: str) -> str:
    if task == "contradiction":
        return """Choose exactly one label:
- CONTRADICTION: incompatible claims under comparable conditions.
- AGREEMENT: compatible conclusions.
- DIFFERENT SCOPE: different populations, tasks, settings, or outcomes that are not directly comparable.
- UNCERTAIN: visible abstract evidence is insufficient.
Return a concise evidence-based reason. Do not infer from anything outside the visible titles and abstracts."""
    if task == "support":
        return """Choose exactly one label:
- SUPPORTED: passage directly entails the complete claim.
- PARTIALLY_SUPPORTED: passage supports a material part but not the complete claim.
- UNSUPPORTED: passage is related or silent but does not justify the claim.
- CONTRADICTED: passage conflicts with the claim.
Judge only the visible claim and passage. A difficult negative is not automatically unsupported."""
    return """Rate every dimension using only 1, 3, or 5:
- evidence: 1 no support; 3 partial support; 5 clear multi-paper support.
- novelty: 1 obvious; 3 some new connection; 5 clearly non-trivial.
- feasibility: 1 not testable; 3 possible with major work; 5 specific and practically testable.
- specificity: 1 vague; 3 partly defined; 5 clear variables and expected relationship.
- usefulness: 1 no clear value; 3 potential value; 5 strong research value.
Use only the visible hypothesis and displayed evidence."""


def _response_example(task: str) -> dict:
    if task == "contradiction":
        return {"label": "AGREEMENT", "reason": "Concise evidence-based reason"}
    if task == "support":
        return {"label": "SUPPORTED", "notes": "Concise evidence-based note"}
    return {
        "evidence": 3, "novelty": 3, "feasibility": 3,
        "specificity": 3, "usefulness": 3, "notes": "Concise note",
    }


def _prompt(task: str, reviewer_id: str, records: list[Mapping[str, Any]]) -> str:
    id_field = ID_FIELDS[task]
    visible = [{key: value for key, value in record.items() if key != "response"} for record in records]
    example = {"annotations": [{"id": "EXAMPLE_ID", "response": _response_example(task)}]}
    return f"""You are producing AI-generated reference annotations, not human judgments.
Use ONLY the visible records below. Do not assume model predictions, hidden labels, confidence, HNS, feasibility metadata, or external facts.
Annotation pass slot: {reviewer_id}. This slot is not an independent human reviewer.

TASK: {task}
{_rubric(task)}

Return strict JSON only in this shape:
{json.dumps(example, ensure_ascii=False)}
The id must be copied exactly from field {id_field}. Return exactly one annotation per input record, no extras.

VISIBLE RECORDS:
{json.dumps(visible, ensure_ascii=False)}"""


def _parse_judge_output(raw: Any, task: str, expected_ids: list[str], model: str, reviewer_id: str) -> dict[str, dict]:
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as error:
            raise Phase3AnnotationError(f"AI judge returned invalid JSON: {error}") from error
    if not isinstance(raw, Mapping) or not isinstance(raw.get("annotations"), list):
        raise Phase3AnnotationError("AI judge output must contain annotations list")
    parsed = {}
    for index, item in enumerate(raw["annotations"]):
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise Phase3AnnotationError(f"AI annotation[{index}] has no ID")
        item_id = item["id"]
        if item_id in parsed:
            raise Phase3AnnotationError(f"Duplicate AI annotation ID: {item_id}")
        response = item.get("response")
        if not isinstance(response, Mapping):
            raise Phase3AnnotationError(f"AI annotation {item_id} has malformed response")
        allowed = ({"label", "reason"} if task == "contradiction" else
                   {"label", "notes"} if task == "support" else
                   {*HYPOTHESIS_DIMENSIONS, "notes"})
        filtered = {key: value for key, value in response.items() if key in allowed}
        filtered.update({
            "timestamp": _now(), "annotation_source": AI_SOURCE,
            "annotation_model": model, "annotation_pass_slot": reviewer_id,
            "independent_human_annotation": False,
        })
        parsed[item_id] = _validate_response(task, filtered, f"AI response[{item_id}]")
    if set(parsed) != set(expected_ids):
        raise Phase3AnnotationError(
            f"AI response IDs differ; expected={sorted(expected_ids)}, got={sorted(parsed)}"
        )
    return parsed


def _judge_validated(
    judge: Judge,
    prompt: str,
    task: str,
    expected_ids: list[str],
    model: str,
    reviewer_id: str,
    *,
    max_attempts: int = 6,
) -> dict[str, dict]:
    last_error = None
    attempt_prompt = prompt
    for _ in range(max_attempts):
        try:
            return _parse_judge_output(
                judge(attempt_prompt), task, expected_ids, model, reviewer_id,
            )
        except Phase3AnnotationError as error:
            last_error = error
            attempt_prompt = (
                f"{prompt}\n\nCORRECTION REQUIRED: Your previous output was invalid: {error}. "
                "Return the complete strict JSON response again, correcting this exact error."
            )
    raise Phase3AnnotationError(f"Could not parse AI batch {expected_ids}: {last_error}")


def annotate_packets(
    packet_dir: str | Path,
    judge: Judge,
    *,
    model: str,
    chunk_sizes: Mapping[str, int] | None = None,
    max_parse_attempts: int = 6,
) -> dict:
    """Fill only null responses, atomically saving after every valid batch."""
    chunks = dict(DEFAULT_CHUNKS if chunk_sizes is None else chunk_sizes)
    summary = {"generated": 0, "preserved": 0, "files": {}}
    for path in sorted(Path(packet_dir).glob("reviewer_*/*.json")):
        packet = _read_json(path)
        validate_reviewer_packet(packet, require_complete=False)
        task = packet["task"]
        id_field = ID_FIELDS[task]
        existing_ids = [record[id_field] for record in packet["records"] if record.get("response") is not None]
        pending = [record for record in packet["records"] if record.get("response") is None]
        summary["preserved"] += len(existing_ids)
        generated_here = 0
        start = 0
        while start < len(pending):
            end = min(start + chunks[task], len(pending))
            batch = pending[start:end]
            prompt = _prompt(task, packet["reviewer_id"], batch)
            while len(prompt) > DEFAULT_MAX_PROMPT_CHARS and len(batch) > 1:
                end -= 1
                batch = pending[start:end]
                prompt = _prompt(task, packet["reviewer_id"], batch)
            if len(prompt) > DEFAULT_MAX_PROMPT_CHARS:
                raise Phase3AnnotationError(
                    f"Single-record AI prompt exceeds {DEFAULT_MAX_PROMPT_CHARS} characters: "
                    f"{path}/{batch[0][id_field]}"
                )
            expected = [record[id_field] for record in batch]
            try:
                decisions = _judge_validated(
                    judge, prompt, task, expected, model, packet["reviewer_id"],
                    max_attempts=max_parse_attempts,
                )
            except Phase3AnnotationError as error:
                raise Phase3AnnotationError(f"Could not parse AI batch {path}/{expected}: {error}") from error
            for record in batch:
                record["response"] = decisions[record[id_field]]
            generated_here += len(batch)
            summary["generated"] += len(batch)
            annotation_models = sorted({
                response["annotation_model"]
                for item in packet["records"]
                if isinstance((response := item.get("response")), Mapping)
                and isinstance(response.get("annotation_model"), str)
            })
            packet["annotation_methodology"] = {
                "annotation_source": AI_SOURCE, "models": annotation_models,
                "independent_human_review": False, "human_ground_truth": False,
                "preexisting_responses_preserved": [
                    item for item in existing_ids if item in PREEXISTING_RESPONSE_IDS
                ],
            }
            packet["status"] = "complete" if all(
                record.get("response") is not None for record in packet["records"]
            ) else "in_progress_ai_annotation"
            _write_json(path, packet, overwrite=True)
            start = end
        validate_reviewer_packet(packet, require_complete=True)
        summary["files"][str(path)] = {
            "generated": generated_here, "preserved": len(existing_ids),
            "total": len(packet["records"]),
        }
    return summary


def find_ai_disagreements(manifest: Mapping[str, Any], judgments: Mapping[str, Mapping[str, dict]]) -> list[dict]:
    assignments = {item["item_id"]: item for item in manifest["assignments"]}
    disagreements = []
    for item_id, assignment in assignments.items():
        supplied = judgments.get(item_id, {})
        if set(supplied) != set(assignment["reviewer_ids"]):
            raise Phase3AnnotationError(f"Responses do not match assignment for {item_id}")
        if len(supplied) == 2 and len({_response_key(assignment["task"], row) for row in supplied.values()}) > 1:
            disagreements.append({
                "item_id": item_id, "task": assignment["task"], "split": assignment["split"],
                "reviewer_ids": assignment["reviewer_ids"],
                "responses": [{"slot": slot, "response": supplied[slot]} for slot in assignment["reviewer_ids"]],
            })
    return disagreements


def _consensus_prompt(task: str, record: Mapping[str, Any], disagreement: Mapping[str, Any]) -> str:
    visible = {key: value for key, value in record.items() if key not in {"annotations", "ratings", "adjudication"}}
    return f"""Produce an AI consensus reference, not human adjudication.
Use only the visible record and the two AI pass responses. Do not use hidden system information.
TASK: {task}
{_rubric(task)}
Return strict JSON only: {json.dumps({'annotations': [{'id': disagreement['item_id'], 'response': _response_example(task)}]})}
VISIBLE RECORD: {json.dumps(visible, ensure_ascii=False)}
AI PASS RESPONSES: {json.dumps(disagreement['responses'], ensure_ascii=False)}"""


def generate_ai_consensus(
    pool_dir: str | Path,
    manifest_path: str | Path,
    packet_dir: str | Path,
    output_path: str | Path,
    judge: Judge,
    *, model: str,
) -> dict:
    pools = _load_pools(pool_dir)
    manifest = _read_json(manifest_path)
    judgments = load_completed_packets(packet_dir)
    disagreements = find_ai_disagreements(manifest, judgments)
    expected_ids = {item["item_id"] for item in disagreements}
    records_by_id = {}
    for (task, _), pool in pools.items():
        for record in pool["records"]:
            records_by_id[record[ID_FIELDS[task]]] = record
    output_file = Path(output_path)
    cached = {}
    if output_file.exists():
        previous = _read_json(output_file)
        for item in previous.get("records", []):
            if not isinstance(item, Mapping) or not isinstance(item.get("item_id"), str):
                raise Phase3AnnotationError("Cached AI consensus records are malformed")
            if item["item_id"] in cached:
                raise Phase3AnnotationError("Cached AI consensus IDs must be unique")
            cached[item["item_id"]] = dict(item)
        if not set(cached).issubset(expected_ids):
            raise Phase3AnnotationError("Cached AI consensus contains unexpected IDs")
    consensus = []
    for disagreement in disagreements:
        item_id, task = disagreement["item_id"], disagreement["task"]
        if item_id in cached:
            cached_item = cached[item_id]
            _validate_response(task, cached_item.get("response"), f"cached consensus[{item_id}]")
            if cached_item.get("is_human_adjudication") is not False:
                raise Phase3AnnotationError(f"Cached consensus {item_id} is not marked non-human")
            consensus.append(cached_item)
        else:
            parsed = _judge_validated(
                judge, _consensus_prompt(task, records_by_id[item_id], disagreement),
                task, [item_id], model, "reviewer_03",
            )[item_id]
            parsed["annotation_source"] = "AI-generated consensus"
            consensus.append({
                **disagreement, "adjudicator_id": "reviewer_03", "response": parsed,
                "is_human_adjudication": False,
            })
            partial = {
                "version": "1.0-ai-reference", "status": "ai_consensus_in_progress",
                "annotation_source": AI_SOURCE, "human_adjudication": False,
                "models": sorted({row["response"]["annotation_model"] for row in consensus}),
                "records": consensus,
            }
            _write_json(output_file, partial, overwrite=True)
    output = {
        "version": "1.0-ai-reference", "status": "ai_consensus_complete",
        "annotation_source": AI_SOURCE, "human_adjudication": False,
        "models": sorted({row["response"]["annotation_model"] for row in consensus}),
        "records": consensus,
    }
    _write_json(output_path, output, overwrite=True)
    return output


def _collect_annotation_models(
    judgments: Mapping[str, Mapping[str, Mapping[str, Any]]],
    consensus_data: Mapping[str, Any],
) -> list[str]:
    models = {
        response["annotation_model"]
        for reviewer_responses in judgments.values()
        for response in reviewer_responses.values()
        if isinstance(response.get("annotation_model"), str)
    }
    for record in consensus_data.get("records", []):
        response = record.get("response", {}) if isinstance(record, Mapping) else {}
        if isinstance(response, Mapping) and isinstance(response.get("annotation_model"), str):
            models.add(response["annotation_model"])
    return sorted(models)


def finalize_ai_references(
    pool_dir: str | Path,
    manifest_path: str | Path,
    packet_dir: str | Path,
    consensus_path: str | Path,
    annotated_pool_dir: str | Path,
    benchmark_dir: str | Path,
    *, model: str,
) -> dict:
    pools = _load_pools(pool_dir)
    manifest = _read_json(manifest_path)
    judgments = load_completed_packets(packet_dir)
    consensus_data = _read_json(consensus_path)
    consensus = {item["item_id"]: item for item in consensus_data.get("records", [])}
    expected = {item["item_id"] for item in find_ai_disagreements(manifest, judgments)}
    if set(consensus) != expected:
        raise Phase3AnnotationError("AI consensus records do not exactly match AI disagreements")
    annotated = build_annotated_pools(pools, manifest, judgments, consensus)
    provenance = {
        "annotation_source": AI_SOURCE,
        "annotation_models": _collect_annotation_models(judgments, consensus_data),
        "consensus_models": consensus_data.get("models", [model]),
        "human_ground_truth": False,
        "independent_human_review": False,
        "human_agreement_calculated": False,
        "human_cohen_kappa_calculated": False,
        "consensus_source": "AI-generated where AI passes differed",
        "preexisting_unconfirmed_non_human_response_ids": PREEXISTING_RESPONSE_IDS,
    }
    filenames = {
        "contradiction": "contradiction_pairs.json",
        "support": "claim_support.json",
        "hypothesis": "hypothesis_ratings.json",
    }
    benchmarks = {}
    for task in ("contradiction", "support", "hypothesis"):
        dev, test = annotated[(task, "dev")], annotated[(task, "test")]
        for pool in (dev, test):
            pool["annotation_provenance"] = provenance
        merged = {**dev, "records": [*dev["records"], *test["records"]]}
        benchmark = finalize_annotation_pool(merged, benchmark_version="1.0-ai-reference-draft")
        benchmark["annotation_provenance"] = provenance
        benchmarks[task] = benchmark
        _write_json(Path(benchmark_dir) / filenames[task], benchmark, overwrite=True)
        for split, pool in (("dev", dev), ("test", test)):
            _write_json(Path(annotated_pool_dir) / POOL_FILES[(task, split)], pool, overwrite=True)
    _write_json(Path(annotated_pool_dir) / "annotation_metadata.json", {
        "phase3_status": "complete_ai_reference_annotations",
        **provenance,
        "counts": {task: len(benchmark[next(key for key in ("pairs", "items", "hypotheses") if key in benchmark)]) for task, benchmark in benchmarks.items()},
    }, overwrite=True)
    return benchmarks


def _groq_judge(model: str) -> Judge:
    from groq import Groq
    from src.utils.config import GROQ_API_KEY, is_configured
    from src.utils.groq_client import groq_chat_with_retry

    if not is_configured(GROQ_API_KEY):
        raise RuntimeError("GROQ_API_KEY is required for AI annotation")
    client = Groq(api_key=GROQ_API_KEY)
    return lambda prompt: groq_chat_with_retry(
        client, prompt, model=model, fallback_model=model,
        max_tokens=2000, temperature=0.0, max_retries=6,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    annotate = sub.add_parser("annotate")
    annotate.add_argument("--packet-dir", default="evaluation/phase3/reviewer_packets")
    annotate.add_argument("--model", default="openai/gpt-oss-20b")
    consensus = sub.add_parser("consensus")
    consensus.add_argument("--pool-dir", default="evaluation/annotation_pools")
    consensus.add_argument("--manifest", default="evaluation/phase3/assignment_manifest.json")
    consensus.add_argument("--packet-dir", default="evaluation/phase3/reviewer_packets")
    consensus.add_argument("--output", default="evaluation/phase3/ai_consensus.json")
    consensus.add_argument("--model", default="openai/gpt-oss-20b")
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--pool-dir", default="evaluation/annotation_pools")
    finalize.add_argument("--manifest", default="evaluation/phase3/assignment_manifest.json")
    finalize.add_argument("--packet-dir", default="evaluation/phase3/reviewer_packets")
    finalize.add_argument("--consensus", default="evaluation/phase3/ai_consensus.json")
    finalize.add_argument("--annotated-pool-dir", default="evaluation/phase3/ai_annotated_pools")
    finalize.add_argument("--benchmark-dir", default="evaluation/benchmarks")
    finalize.add_argument("--model", default="openai/gpt-oss-20b")
    args = parser.parse_args(argv)
    if args.command == "annotate":
        result = annotate_packets(args.packet_dir, _groq_judge(args.model), model=args.model)
    elif args.command == "consensus":
        result = generate_ai_consensus(
            args.pool_dir, args.manifest, args.packet_dir, args.output,
            _groq_judge(args.model), model=args.model,
        )
    else:
        result = finalize_ai_references(
            args.pool_dir, args.manifest, args.packet_dir, args.consensus,
            args.annotated_pool_dir, args.benchmark_dir, model=args.model,
        )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
