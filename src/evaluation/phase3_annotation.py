"""Reviewer-isolated Phase 3 annotation workflow for reasoning benchmarks.

This module reads only blinded annotation pools. It never reads protected system
sidecars, and it never derives human labels from model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evaluation.annotation_pool import (
    BLINDED_KEYS,
    AnnotationPoolError,
    finalize_annotation_pool,
    validate_annotation_pool,
)
from src.evaluation.reasoning_benchmark_io import (
    CONTRADICTION_LABELS,
    HYPOTHESIS_DIMENSIONS,
    HYPOTHESIS_SCORES,
    SUPPORT_LABELS,
)


POOL_FILES = {
    ("contradiction", "dev"): "contradiction_dev.json",
    ("contradiction", "test"): "contradiction_test.json",
    ("support", "dev"): "claim_support_dev.json",
    ("support", "test"): "claim_support_test.json",
    ("hypothesis", "dev"): "hypothesis_dev.json",
    ("hypothesis", "test"): "hypothesis_test.json",
}
ID_FIELDS = {
    "contradiction": "pair_id",
    "support": "item_id",
    "hypothesis": "hypothesis_id",
}
HUMAN_FIELDS = {
    "contradiction": "annotations",
    "support": "annotations",
    "hypothesis": "ratings",
}
LABELS = {"contradiction": CONTRADICTION_LABELS, "support": SUPPORT_LABELS}
REVIEWER_ID_RE = re.compile(r"^reviewer_[0-9]{2,}$")


class Phase3AnnotationError(ValueError):
    """Raised when Phase 3 annotation integrity checks fail."""


def _read_json(path: str | Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase3AnnotationError(f"Could not load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase3AnnotationError(f"{path} must contain a JSON object")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any], *, overwrite: bool = False) -> None:
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def _reviewer_id(value: Any, context: str) -> str:
    if not isinstance(value, str) or not REVIEWER_ID_RE.fullmatch(value):
        raise Phase3AnnotationError(
            f"{context} must be anonymized as reviewer_ followed by at least two digits"
        )
    return value


def _timestamp(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Phase3AnnotationError(f"{context} must be a non-empty ISO-8601 timestamp")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Phase3AnnotationError(f"{context} is not ISO-8601") from error
    return value


def _stable_rank(seed: str, task: str, item_id: str) -> str:
    return hashlib.sha256(f"{seed}\x1f{task}\x1f{item_id}".encode()).hexdigest()


def _load_pools(pool_dir: str | Path) -> dict[tuple[str, str], dict]:
    root = Path(pool_dir)
    pools = {}
    for key, filename in POOL_FILES.items():
        pool = _read_json(root / filename)
        validate_annotation_pool(pool, key[0])
        if any(record["split"] != key[1] for record in pool["records"]):
            raise Phase3AnnotationError(f"{filename} contains the wrong split")
        pools[key] = pool
    return pools


def _assert_frozen_references(
    pools: Mapping[tuple[str, str], Mapping[str, Any]],
    corpus_path: str | Path,
    query_benchmark_path: str | Path,
) -> None:
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    if not isinstance(corpus, list):
        raise Phase3AnnotationError("Frozen corpus must be a JSON list")
    paper_ids = {
        str(paper.get("id") or paper.get("paperId") or "").strip()
        for paper in corpus if isinstance(paper, Mapping)
    }
    query_data = _read_json(query_benchmark_path)
    query_ids = {
        item.get("query_id") for item in query_data.get("queries", [])
        if isinstance(item, Mapping)
    }
    for (task, split), pool in pools.items():
        for record in pool["records"]:
            item_id = record[ID_FIELDS[task]]
            if task == "contradiction":
                references = [record["paper1_id"], record["paper2_id"]]
            elif task == "support":
                references = [record["paper_id"]]
            else:
                references = [
                    evidence.get("paper_id") for evidence in record["evidence"]
                    if isinstance(evidence, Mapping) and evidence.get("paper_id")
                ]
            unknown = sorted(set(references).difference(paper_ids))
            if unknown:
                raise Phase3AnnotationError(f"{item_id} has unknown frozen paper IDs: {unknown}")
            if task != "contradiction" and record["query_id"] not in query_ids:
                raise Phase3AnnotationError(f"{item_id} has unknown frozen query ID")


def _display_record(task: str, record: Mapping[str, Any]) -> dict:
    excluded = {HUMAN_FIELDS[task], "adjudication"}
    display = {key: value for key, value in record.items() if key not in excluded}
    leaks = _find_leaks(display)
    if leaks:
        raise Phase3AnnotationError(f"Annotator display leaks protected fields: {leaks}")
    return display


def _find_leaks(value: Any, path: str = "record") -> list[str]:
    leaks = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in BLINDED_KEYS:
                leaks.append(child_path)
            leaks.extend(_find_leaks(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            leaks.extend(_find_leaks(child, f"{path}[{index}]"))
    return leaks


def _response_schema(task: str) -> dict:
    if task == "contradiction":
        return {"label": sorted(CONTRADICTION_LABELS), "reason": "required", "timestamp": "ISO-8601"}
    if task == "support":
        return {"label": sorted(SUPPORT_LABELS), "notes": "optional", "timestamp": "ISO-8601"}
    return {
        "scores": {dimension: sorted(HYPOTHESIS_SCORES) for dimension in HYPOTHESIS_DIMENSIONS},
        "notes": "optional", "timestamp": "ISO-8601",
    }


def prepare_annotation_workflow(
    pool_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: str,
    double_fraction: float = 0.25,
    reviewer_ids: tuple[str, str] = ("reviewer_01", "reviewer_02"),
    adjudicator_id: str = "reviewer_03",
    corpus_path: str | Path | None = None,
    query_benchmark_path: str | Path | None = None,
) -> dict:
    """Create deterministic assignments and isolated, initially empty packets."""
    if not isinstance(seed, str) or not seed.strip():
        raise Phase3AnnotationError("seed must be non-empty")
    if not 0 < double_fraction <= 1:
        raise Phase3AnnotationError("double_fraction must be in (0, 1]")
    reviewers = tuple(_reviewer_id(item, "reviewer_id") for item in reviewer_ids)
    adjudicator = _reviewer_id(adjudicator_id, "adjudicator_id")
    if len(set((*reviewers, adjudicator))) != 3:
        raise Phase3AnnotationError("Reviewer and adjudicator IDs must be distinct")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 3 directory: {output}")
    pools = _load_pools(pool_dir)
    if corpus_path and query_benchmark_path:
        _assert_frozen_references(pools, corpus_path, query_benchmark_path)

    assignments = []
    packets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    task_counts = {}
    for task in ("contradiction", "support", "hypothesis"):
        records = [record for split in ("dev", "test") for record in pools[(task, split)]["records"]]
        id_field = ID_FIELDS[task]
        ranked = sorted(records, key=lambda item: _stable_rank(seed, task, item[id_field]))
        double_count = math.ceil(len(records) * double_fraction)
        double_ids = {item[id_field] for item in ranked[:double_count]}
        task_counts[task] = {
            "total": len(records), "double_annotated": double_count,
            "double_fraction_actual": round(double_count / len(records), 6),
        }
        for record in records:
            item_id = record[id_field]
            primary_index = int(_stable_rank(seed + "-primary", task, item_id), 16) % 2
            assigned = list(reviewers) if item_id in double_ids else [reviewers[primary_index]]
            assignments.append({
                "task": task, "split": record["split"], "item_id": item_id,
                "reviewer_ids": assigned, "double_annotation": len(assigned) == 2,
            })
            for reviewer in assigned:
                packets[(reviewer, task, record["split"])].append({
                    **_display_record(task, record), "response": None,
                })

    manifest = {
        "phase3_version": "1.0-draft", "status": "annotation_ready",
        "seed": seed, "double_annotation_fraction_requested": double_fraction,
        "reviewer_slots": [
            {"reviewer_id": reviewers[0], "role": "reviewer", "assigned_person": None},
            {"reviewer_id": reviewers[1], "role": "reviewer", "assigned_person": None},
            {"reviewer_id": adjudicator, "role": "adjudicator", "assigned_person": None},
        ],
        "task_counts": task_counts,
        "assignments": sorted(assignments, key=lambda item: (item["task"], item["split"], item["item_id"])),
    }
    _write_json(output / "assignment_manifest.json", manifest)
    for (reviewer, task, split), records in packets.items():
        packet = {
            "packet_version": "1.0", "status": "unstarted", "blinded": True,
            "reviewer_id": reviewer, "task": task, "split": split,
            "response_schema": _response_schema(task), "records": records,
        }
        validate_reviewer_packet(packet, require_complete=False)
        _write_json(output / "reviewer_packets" / reviewer / f"{task}_{split}.json", packet)
    _write_json(output / "adjudication_template.json", {
        "version": "1.0", "status": "pending_human_responses",
        "adjudicator_id": adjudicator, "records": [],
    })
    return manifest


def _validate_response(task: str, response: Any, context: str) -> dict:
    if not isinstance(response, Mapping):
        raise Phase3AnnotationError(f"{context} must be an object")
    result = dict(response)
    if task in LABELS:
        if result.get("label") not in LABELS[task]:
            raise Phase3AnnotationError(f"{context}.label is invalid")
        text_field = "reason" if task == "contradiction" else "notes"
        value = result.get(text_field, "")
        if not isinstance(value, str) or (task == "contradiction" and not value.strip()):
            raise Phase3AnnotationError(f"{context}.{text_field} is invalid")
        result[text_field] = value.strip()
    else:
        for dimension in HYPOTHESIS_DIMENSIONS:
            if result.get(dimension) not in HYPOTHESIS_SCORES:
                raise Phase3AnnotationError(f"{context}.{dimension} must be 1, 3, or 5")
        if not isinstance(result.get("notes", ""), str):
            raise Phase3AnnotationError(f"{context}.notes must be a string")
        result["notes"] = result.get("notes", "").strip()
    _timestamp(result.get("timestamp"), f"{context}.timestamp")
    return result


def validate_reviewer_packet(packet: Mapping[str, Any], *, require_complete: bool) -> None:
    reviewer = _reviewer_id(packet.get("reviewer_id"), "packet.reviewer_id")
    task = packet.get("task")
    split = packet.get("split")
    if task not in ID_FIELDS or split not in {"dev", "test"} or packet.get("blinded") is not True:
        raise Phase3AnnotationError("Malformed reviewer packet metadata")
    records = packet.get("records")
    if not isinstance(records, list):
        raise Phase3AnnotationError("packet.records must be a list")
    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise Phase3AnnotationError(f"packet.records[{index}] must be an object")
        item_id = record.get(ID_FIELDS[task])
        if not isinstance(item_id, str) or item_id in seen:
            raise Phase3AnnotationError("Packet IDs must be non-empty and unique")
        seen.add(item_id)
        display = {key: value for key, value in record.items() if key != "response"}
        leaks = _find_leaks(display)
        if leaks:
            raise Phase3AnnotationError(f"Packet leaks system fields: {leaks}")
        response = record.get("response")
        if response is None:
            if require_complete:
                raise Phase3AnnotationError(f"{reviewer} has no response for {item_id}")
        else:
            _validate_response(task, response, f"response[{item_id}]")


def load_completed_packets(packet_dir: str | Path) -> dict[str, dict[str, dict]]:
    judgments: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(Path(packet_dir).glob("reviewer_*/*.json")):
        packet = _read_json(path)
        validate_reviewer_packet(packet, require_complete=True)
        task = packet["task"]
        id_field = ID_FIELDS[task]
        reviewer = packet["reviewer_id"]
        for record in packet["records"]:
            item_id = record[id_field]
            if reviewer in judgments[item_id]:
                raise Phase3AnnotationError(f"Duplicate response from {reviewer} for {item_id}")
            judgments[item_id][reviewer] = dict(record["response"])
    return dict(judgments)


def _response_key(task: str, response: Mapping[str, Any]) -> tuple:
    if task in LABELS:
        return (response["label"],)
    return tuple(response[dimension] for dimension in HYPOTHESIS_DIMENSIONS)


def _cohen_kappa(values: list[tuple[Any, Any]]) -> float | None:
    if not values:
        return None
    agreement = sum(first == second for first, second in values) / len(values)
    first_counts = Counter(first for first, _ in values)
    second_counts = Counter(second for _, second in values)
    categories = set(first_counts) | set(second_counts)
    expected = sum(first_counts[item] * second_counts[item] for item in categories) / len(values) ** 2
    if expected == 1:
        return None
    return round((agreement - expected) / (1 - expected), 6)


def analyze_responses(manifest: Mapping[str, Any], judgments: Mapping[str, Mapping[str, dict]]) -> tuple[dict, dict]:
    expected = {item["item_id"]: item for item in manifest["assignments"]}
    if set(judgments) != set(expected):
        missing = sorted(set(expected).difference(judgments))
        extra = sorted(set(judgments).difference(expected))
        raise Phase3AnnotationError(f"Response IDs differ; missing={missing}, extra={extra}")
    task_pairs: dict[str, list[tuple]] = defaultdict(list)
    queue = []
    for item_id, assignment in expected.items():
        supplied = judgments[item_id]
        if set(supplied) != set(assignment["reviewer_ids"]):
            raise Phase3AnnotationError(f"Reviewer responses do not match assignment for {item_id}")
        if assignment["double_annotation"]:
            reviewers = assignment["reviewer_ids"]
            first, second = supplied[reviewers[0]], supplied[reviewers[1]]
            keys = (_response_key(assignment["task"], first), _response_key(assignment["task"], second))
            task_pairs[assignment["task"]].append(keys)
            if keys[0] != keys[1]:
                queue.append({
                    "item_id": item_id, "task": assignment["task"], "split": assignment["split"],
                    "reviewer_ids": reviewers,
                    "original_responses": [{"reviewer_id": reviewer, **supplied[reviewer]} for reviewer in reviewers],
                    "adjudication": None,
                })
    agreement = {"status": "human_responses_analyzed", "tasks": {}}
    for task in ("contradiction", "support", "hypothesis"):
        pairs = task_pairs[task]
        agreeing = sum(first == second for first, second in pairs)
        entry = {
            "double_annotated": len(pairs), "agreeing": agreeing,
            "disagreeing": len(pairs) - agreeing,
            "agreement_rate": round(agreeing / len(pairs), 6) if pairs else None,
        }
        if task in LABELS:
            entry["cohen_kappa"] = _cohen_kappa([(a[0], b[0]) for a, b in pairs])
        else:
            entry["cohen_kappa_by_dimension"] = {
                dimension: _cohen_kappa([(a[index], b[index]) for a, b in pairs])
                for index, dimension in enumerate(HYPOTHESIS_DIMENSIONS)
            }
        agreement["tasks"][task] = entry
    return agreement, {"version": "1.0", "status": "awaiting_adjudication", "records": queue}


def build_annotated_pools(
    pools: Mapping[tuple[str, str], Mapping[str, Any]],
    manifest: Mapping[str, Any],
    judgments: Mapping[str, Mapping[str, dict]],
    adjudications: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], dict]:
    assignment_by_id = {item["item_id"]: item for item in manifest["assignments"]}
    output = json.loads(json.dumps({f"{task}:{split}": pool for (task, split), pool in pools.items()}))
    result = {}
    for key, pool in output.items():
        task, split = key.split(":")
        for record in pool["records"]:
            item_id = record[ID_FIELDS[task]]
            assignment = assignment_by_id[item_id]
            responses = judgments[item_id]
            human = []
            for reviewer in assignment["reviewer_ids"]:
                response = dict(responses[reviewer])
                response["reviewer_id"] = reviewer
                human.append(response)
            record[HUMAN_FIELDS[task]] = human
            keys = {_response_key(task, response) for response in responses.values()}
            if len(keys) > 1:
                adjudication = adjudications.get(item_id)
                if adjudication is None:
                    raise Phase3AnnotationError(f"Unresolved disagreement for {item_id}")
                adjudicator = _reviewer_id(adjudication.get("adjudicator_id"), "adjudicator_id")
                final_response = _validate_response(task, adjudication.get("response"), f"adjudication[{item_id}]")
                record["adjudication"] = {
                    "reviewer_id": adjudicator, **final_response,
                    "reviewer_ids": assignment["reviewer_ids"],
                    "original_responses": human,
                }
            else:
                record["adjudication"] = None
        result[(task, split)] = pool
    return result


def analyze_phase3(
    manifest_path: str | Path,
    packet_dir: str | Path,
    output_dir: str | Path,
) -> tuple[dict, dict]:
    """Validate complete independent responses and create human-only review files."""
    manifest = _read_json(manifest_path)
    judgments = load_completed_packets(packet_dir)
    agreement, queue = analyze_responses(manifest, judgments)
    adjudicator = next(
        slot["reviewer_id"] for slot in manifest["reviewer_slots"]
        if slot["role"] == "adjudicator"
    )
    adjudications = {
        "version": "1.0", "status": "awaiting_adjudication",
        "records": [{
            **record, "adjudicator_id": adjudicator, "response": None,
        } for record in queue["records"]],
    }
    output = Path(output_dir)
    _write_json(output / "agreement_report.json", agreement)
    _write_json(output / "adjudications.json", adjudications)
    return agreement, adjudications


def finalize_phase3(
    pool_dir: str | Path,
    manifest_path: str | Path,
    packet_dir: str | Path,
    adjudication_path: str | Path,
    output_dir: str | Path,
    benchmark_dir: str | Path,
    *, benchmark_version: str = "1.0-draft",
) -> dict:
    """Finalize complete human work to annotated pools and draft benchmarks."""
    pools = _load_pools(pool_dir)
    manifest = _read_json(manifest_path)
    judgments = load_completed_packets(packet_dir)
    agreement, queue = analyze_responses(manifest, judgments)
    adjudication_data = _read_json(adjudication_path)
    adjudications = {
        item["item_id"]: item for item in adjudication_data.get("records", [])
        if isinstance(item, Mapping) and isinstance(item.get("item_id"), str)
    }
    if len(adjudications) != len(adjudication_data.get("records", [])):
        raise Phase3AnnotationError("Adjudication IDs must be present and unique")
    expected_disagreements = {item["item_id"] for item in queue["records"]}
    if set(adjudications) != expected_disagreements:
        raise Phase3AnnotationError("Adjudications must exactly match detected disagreements")
    annotated = build_annotated_pools(pools, manifest, judgments, adjudications)
    combined = {}
    benchmark_paths = {
        "contradiction": "contradiction_pairs.json",
        "support": "claim_support.json",
        "hypothesis": "hypothesis_ratings.json",
    }
    for task in ("contradiction", "support", "hypothesis"):
        dev, test = annotated[(task, "dev")], annotated[(task, "test")]
        merged = {**dev, "records": [*dev["records"], *test["records"]]}
        benchmark = finalize_annotation_pool(merged, benchmark_version=benchmark_version)
        combined[task] = benchmark
        _write_json(Path(benchmark_dir) / benchmark_paths[task], benchmark, overwrite=True)
        for split, pool in (("dev", dev), ("test", test)):
            _write_json(Path(output_dir) / POOL_FILES[(task, split)], pool, overwrite=True)
    _write_json(Path(output_dir) / "agreement_report.json", agreement, overwrite=True)
    return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--pool-dir", default="evaluation/annotation_pools")
    prepare.add_argument("--output-dir", default="evaluation/phase3")
    prepare.add_argument("--seed", default="phase3-double-annotation-v1")
    prepare.add_argument("--double-fraction", type=float, default=0.25)
    prepare.add_argument("--corpus", default="data/s2_clean.json")
    prepare.add_argument("--queries", default="evaluation/benchmarks/retrieval_queries_judged.json")
    validate = subparsers.add_parser("validate-packet")
    validate.add_argument("--packet", required=True)
    validate.add_argument("--require-complete", action="store_true")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--manifest", default="evaluation/phase3/assignment_manifest.json")
    analyze.add_argument("--packet-dir", default="evaluation/phase3/reviewer_packets")
    analyze.add_argument("--output-dir", default="evaluation/phase3/human_review")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--pool-dir", default="evaluation/annotation_pools")
    finalize.add_argument("--manifest", default="evaluation/phase3/assignment_manifest.json")
    finalize.add_argument("--packet-dir", default="evaluation/phase3/reviewer_packets")
    finalize.add_argument("--adjudications", default="evaluation/phase3/adjudications.json")
    finalize.add_argument("--output-dir", default="evaluation/phase3/annotated_pools")
    finalize.add_argument("--benchmark-dir", default="evaluation/benchmarks")
    arguments = parser.parse_args(argv)
    if arguments.command == "prepare":
        manifest = prepare_annotation_workflow(
            arguments.pool_dir, arguments.output_dir, seed=arguments.seed,
            double_fraction=arguments.double_fraction, corpus_path=arguments.corpus,
            query_benchmark_path=arguments.queries,
        )
        print(json.dumps(manifest["task_counts"], indent=2))
    elif arguments.command == "validate-packet":
        validate_reviewer_packet(_read_json(arguments.packet), require_complete=arguments.require_complete)
        print("Reviewer packet validation passed")
    elif arguments.command == "analyze":
        agreement, adjudications = analyze_phase3(
            arguments.manifest, arguments.packet_dir, arguments.output_dir,
        )
        print(json.dumps({
            "agreement": agreement["tasks"],
            "disagreements": len(adjudications["records"]),
        }, indent=2))
    else:
        finalize_phase3(
            arguments.pool_dir, arguments.manifest, arguments.packet_dir,
            arguments.adjudications, arguments.output_dir, arguments.benchmark_dir,
        )
        print("Draft reasoning benchmarks finalized from human annotations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
