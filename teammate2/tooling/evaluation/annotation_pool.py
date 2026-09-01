"""Generate blinded Teammate 2 annotation pools without inventing labels.

The annotation file contains source evidence and empty human-owned fields. A
separate sidecar contains system predictions and scores. Stable IDs are derived
from immutable source identities, never from predictions or labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.evaluation.reasoning_benchmark_io import (
    CONTRADICTION_LABELS,
    HYPOTHESIS_DIMENSIONS,
    SUPPORT_LABELS,
    VALID_SPLITS,
    validate_claim_support_benchmark,
    validate_contradiction_benchmark,
    validate_hypothesis_benchmark,
)


TASKS = ("contradiction", "support", "hypothesis")
BLINDED_KEYS = {
    "label", "prediction", "predicted_label", "support_label", "verdict",
    "verdict_valid", "confidence", "llm_analysis", "model_feasibility",
    "feasibility", "accepted", "accepted_contradiction", "hns", "candidate_score",
    "concept_jaccard", "year_gap", "llm_analysis", "verdict_valid",
    "retrieval_score", "retrieval_source", "source_paper_rank", "grounded",
    "rejection_reasons", "validation_valid", "evidence_score",
    "concept_overlap", "query_support_ratio", "raw_generation",
    "generation_configuration",
}


class AnnotationPoolError(ValueError):
    """Raised for unsafe, ambiguous, or malformed annotation-pool data."""


def stable_example_id(prefix: str, *parts: str) -> str:
    if not prefix or not parts or any(not isinstance(part, str) or not part.strip() for part in parts):
        raise AnnotationPoolError("Stable IDs require a prefix and non-empty string parts")
    normalized = "\x1f".join(" ".join(part.split()).casefold() for part in parts)
    return f"{prefix}{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:12].upper()}"


def _text(record: Mapping[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise AnnotationPoolError(f"{context}.{field} must be a non-empty string")
    return value.strip()


def _split(record: Mapping[str, Any], default_split: str | None, context: str) -> str:
    split = record.get("split", default_split)
    if split not in VALID_SPLITS:
        raise AnnotationPoolError(f"{context}.split must be one of {sorted(VALID_SPLITS)}")
    return split


def _paper(record: Mapping[str, Any], number: int, context: str) -> dict:
    nested = record.get(f"paper{number}")
    if nested is not None and not isinstance(nested, Mapping):
        raise AnnotationPoolError(f"{context}.paper{number} must be an object")
    if isinstance(nested, Mapping):
        source = nested
        identifier = "id"
    else:
        source = record
        identifier = f"paper{number}_id"
    return {
        "id": _text(source, identifier, context),
        "title": _text(source, "title" if nested is not None else f"paper{number}_title", context),
        "abstract": _text(source, "abstract" if nested is not None else f"paper{number}_abstract", context),
    }


def _ensure_unique(identity, seen: set, context: str) -> None:
    if identity in seen:
        raise AnnotationPoolError(f"Duplicate annotation example: {context}")
    seen.add(identity)


def _prediction_sidecar(task: str, example_id: str, source: Mapping[str, Any]) -> dict:
    sidecar = {
        "example_id": example_id,
        "query_id": source.get("query_id"),
        "candidate_source": source.get("candidate_source"),
        "generation_configuration": source.get("generation_configuration"),
    }
    if task == "contradiction":
        sidecar.update({
            "prediction": source.get("prediction", source.get("verdict")),
            "confidence": source.get("confidence"),
            "candidate_score": source.get("candidate_score"),
            "shared_concepts": source.get("shared_concepts"),
            "shared_concept_names": source.get("shared_concept_names"),
            "concept_jaccard": source.get("concept_jaccard"),
            "year_gap": source.get("year_gap"),
            "flag": source.get("flag"),
            "llm_analysis": source.get("llm_analysis"),
            "verdict_valid": source.get("verdict_valid"),
            "accepted_contradiction": source.get("accepted_contradiction"),
        })
    elif task == "support":
        sidecar.update({
            "prediction": source.get("prediction", source.get("support_label")),
            "confidence": source.get("confidence"),
            "model": source.get("model"),
            "retrieval_score": source.get("retrieval_score"),
            "retrieval_source": source.get("retrieval_source"),
            "source_paper_rank": source.get("source_paper_rank"),
            "claim_grounded": source.get("claim_grounded"),
            "rejection_reasons": source.get("rejection_reasons"),
            "invalid_passage_ids": source.get("invalid_passage_ids"),
        })
    else:
        sidecar.update({
            "model_feasibility": source.get("model_feasibility", source.get("feasibility")),
            "accepted": source.get("accepted"),
            "hns": source.get("hns"),
            "candidate_paper_id": source.get("candidate_paper_id"),
            "candidate_score": source.get("candidate_score"),
            "concept_overlap": source.get("concept_overlap"),
            "query_support_ratio": source.get("query_support_ratio"),
            "validation_valid": source.get("validation_valid"),
            "raw_generation": source.get("raw_generation"),
            "missing_evidence": source.get("missing_evidence"),
        })
    return sidecar


def _find_blinding_leaks(value: Any, path: str = "record") -> list[str]:
    leaks = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in {"annotations", "ratings", "adjudication"}:
                continue
            if key in BLINDED_KEYS:
                leaks.append(child_path)
            leaks.extend(_find_blinding_leaks(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            leaks.extend(_find_blinding_leaks(child, f"{path}[{index}]"))
    return leaks


def generate_annotation_pool(
    task: str,
    candidates: Iterable[Mapping[str, Any]],
    *,
    split: str | None = None,
    fixture_only: bool = False,
    presentation_seed: str | None = None,
) -> tuple[dict, dict]:
    """Return a blinded annotation pool and a separate system-only sidecar."""
    if task not in TASKS:
        raise AnnotationPoolError(f"Unknown task: {task}")
    records = []
    system_records = []
    seen = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise AnnotationPoolError(f"candidate[{index}] must be an object")
        context = f"candidate[{index}]"
        record_split = _split(candidate, split, context)
        if task == "contradiction":
            first = _paper(candidate, 1, context)
            second = _paper(candidate, 2, context)
            if first["id"] == second["id"]:
                raise AnnotationPoolError(f"{context} repeats the same paper")
            if second["id"] < first["id"]:
                first, second = second, first
            identity = (first["id"], second["id"])
            _ensure_unique(identity, seen, context)
            example_id = stable_example_id("C", *identity)
            annotation = {
                "pair_id": example_id,
                "split": record_split,
                "paper1_id": first["id"], "paper1_title": first["title"],
                "paper1_abstract": first["abstract"],
                "paper2_id": second["id"], "paper2_title": second["title"],
                "paper2_abstract": second["abstract"],
                "annotations": [], "adjudication": None,
            }
        elif task == "support":
            query_id = _text(candidate, "query_id", context)
            claim = _text(candidate, "claim", context)
            passage_id = _text(candidate, "passage_id", context)
            identity = (query_id, " ".join(claim.split()).casefold(), passage_id)
            _ensure_unique(identity, seen, context)
            example_id = stable_example_id("S", *identity)
            annotation = {
                "item_id": example_id, "split": record_split, "query_id": query_id,
                "claim": claim, "passage_id": passage_id,
                "passage_text": _text(candidate, "passage_text", context),
                "paper_id": _text(candidate, "paper_id", context),
                "annotations": [], "adjudication": None,
            }
        else:
            query_id = _text(candidate, "query_id", context)
            hypothesis = _text(candidate, "hypothesis", context)
            identity = (query_id, " ".join(hypothesis.split()).casefold())
            _ensure_unique(identity, seen, context)
            example_id = stable_example_id("H", *identity)
            evidence = candidate.get("evidence", [])
            if not isinstance(evidence, list):
                raise AnnotationPoolError(f"{context}.evidence must be a list")
            annotation = {
                "hypothesis_id": example_id, "split": record_split,
                "query_id": query_id, "hypothesis": hypothesis,
                "evidence": evidence, "ratings": [], "adjudication": None,
            }
        records.append(annotation)
        system_records.append(_prediction_sidecar(task, example_id, candidate))

    records.sort(key=lambda item: next(item[key] for key in ("pair_id", "item_id", "hypothesis_id") if key in item))
    system_records.sort(key=lambda item: item["example_id"])
    if task == "hypothesis" and presentation_seed is not None:
        if not isinstance(presentation_seed, str) or not presentation_seed.strip():
            raise AnnotationPoolError("presentation_seed must be a non-empty string")
        random.Random(presentation_seed).shuffle(records)
    pool = {
        "pool_version": "1.0-test-fixture" if fixture_only else "1.0",
        "task": task,
        "fixture_only": fixture_only,
        "blinded": True,
        "presentation_order_randomized": task == "hypothesis" and presentation_seed is not None,
        "records": records,
    }
    sidecar = {
        "pool_version": pool["pool_version"], "task": task,
        "fixture_only": fixture_only, "annotation_blinded": True,
        "system_records": system_records,
    }
    validate_annotation_pool(pool, task)
    return pool, sidecar


def validate_annotation_pool(pool: Mapping[str, Any], task: str) -> None:
    if task not in TASKS or pool.get("task") != task:
        raise AnnotationPoolError("Pool task does not match the requested task")
    if pool.get("blinded") is not True:
        raise AnnotationPoolError("Annotation pools must be marked blinded")
    if not isinstance(pool.get("fixture_only"), bool):
        raise AnnotationPoolError("fixture_only must be boolean")
    records = pool.get("records")
    if not isinstance(records, list):
        raise AnnotationPoolError("records must be a list")
    seen_ids = set()
    seen_pairs = set()
    id_field = {"contradiction": "pair_id", "support": "item_id", "hypothesis": "hypothesis_id"}[task]
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise AnnotationPoolError(f"records[{index}] must be an object")
        leaks = _find_blinding_leaks(record, f"records[{index}]")
        if leaks:
            raise AnnotationPoolError(f"records[{index}] leaks system fields: {sorted(leaks)}")
        identifier = _text(record, id_field, f"records[{index}]")
        if identifier in seen_ids:
            raise AnnotationPoolError(f"Duplicate annotation ID: {identifier}")
        seen_ids.add(identifier)
        _split(record, None, f"records[{index}]")
        if task == "contradiction":
            paper1 = _text(record, "paper1_id", f"records[{index}]")
            paper2 = _text(record, "paper2_id", f"records[{index}]")
            for field in ("paper1_title", "paper1_abstract", "paper2_title", "paper2_abstract"):
                _text(record, field, f"records[{index}]")
            if paper1 >= paper2:
                raise AnnotationPoolError("Contradiction paper IDs must be canonical")
            pair = (paper1, paper2)
            _ensure_unique(pair, seen_pairs, f"records[{index}]")
        elif task == "support":
            for field in ("query_id", "claim", "passage_id", "passage_text", "paper_id"):
                _text(record, field, f"records[{index}]")
        else:
            for field in ("query_id", "hypothesis"):
                _text(record, field, f"records[{index}]")
            if not isinstance(record.get("evidence"), list):
                raise AnnotationPoolError(f"records[{index}].evidence must be a list")
        human_field = "ratings" if task == "hypothesis" else "annotations"
        if not isinstance(record.get(human_field), list):
            raise AnnotationPoolError(f"records[{index}].{human_field} must be a list")


def _adjudicated_annotation(record: Mapping[str, Any], valid_labels: set[str], context: str) -> dict:
    annotations = record.get("annotations", [])
    if not annotations:
        raise AnnotationPoolError(f"{context} has no human annotations")
    for annotation in annotations:
        if not isinstance(annotation, Mapping) or annotation.get("label") not in valid_labels:
            raise AnnotationPoolError(f"{context} contains a malformed human annotation")
        _text(annotation, "reviewer_id", context)
    adjudication = record.get("adjudication")
    if adjudication is not None:
        if not isinstance(adjudication, Mapping) or adjudication.get("label") not in valid_labels:
            raise AnnotationPoolError(f"{context} contains malformed adjudication")
        return dict(adjudication)
    labels = {annotation["label"] for annotation in annotations}
    if len(labels) != 1:
        raise AnnotationPoolError(f"{context} has disagreement but no adjudication")
    return dict(annotations[0])


def finalize_annotation_pool(pool: Mapping[str, Any], *, benchmark_version: str = "1.0-draft") -> dict:
    """Convert completed human annotations into a validated draft benchmark."""
    task = pool.get("task")
    validate_annotation_pool(pool, task)
    records = pool["records"]
    if task == "contradiction":
        pairs = []
        for index, record in enumerate(records):
            decision = _adjudicated_annotation(record, CONTRADICTION_LABELS, f"records[{index}]")
            pairs.append({
                "pair_id": record["pair_id"], "split": record["split"],
                "paper1_id": record["paper1_id"], "paper2_id": record["paper2_id"],
                "label": decision["label"], "reason": _text(decision, "reason", f"records[{index}]") ,
                "annotators": [annotation["reviewer_id"] for annotation in record["annotations"]],
                "adjudicated": record.get("adjudication") is not None,
            })
        benchmark = {
            "benchmark_version": benchmark_version, "status": "draft",
            "fixture_only": pool.get("fixture_only", False), "pairs": pairs,
        }
        validate_contradiction_benchmark(benchmark)
    elif task == "support":
        items = []
        for index, record in enumerate(records):
            decision = _adjudicated_annotation(record, SUPPORT_LABELS, f"records[{index}]")
            items.append({
                **{key: record[key] for key in (
                    "item_id", "split", "query_id", "claim", "passage_id", "passage_text", "paper_id"
                )},
                "label": decision["label"], "notes": str(decision.get("notes") or ""),
            })
        benchmark = {
            "benchmark_version": benchmark_version, "status": "draft",
            "fixture_only": pool.get("fixture_only", False), "items": items,
        }
        validate_claim_support_benchmark(benchmark)
    else:
        hypotheses = []
        for index, record in enumerate(records):
            if not record.get("ratings"):
                raise AnnotationPoolError(f"records[{index}] has no human ratings")
            hypotheses.append({
                "hypothesis_id": record["hypothesis_id"], "split": record["split"],
                "query_id": record["query_id"], "hypothesis": record["hypothesis"],
                "ratings": record["ratings"], "adjudication": record.get("adjudication"),
            })
        benchmark = {
            "benchmark_version": benchmark_version, "status": "draft",
            "fixture_only": pool.get("fixture_only", False), "hypotheses": hypotheses,
        }
        validate_hypothesis_benchmark(benchmark)
    return benchmark


def _load_candidates(path: str | Path) -> list[dict]:
    source = Path(path)
    try:
        if source.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnnotationPoolError(f"Could not load candidates: {error}") from error
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("records"), list):
        return value["records"]
    raise AnnotationPoolError("Candidate input must be a JSON list or an object with records")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--annotation-output", required=True)
    parser.add_argument("--system-output", required=True)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS))
    parser.add_argument("--presentation-seed")
    arguments = parser.parse_args(argv)
    pool, sidecar = generate_annotation_pool(
        arguments.task, _load_candidates(arguments.input), split=arguments.split,
        presentation_seed=arguments.presentation_seed,
    )
    outputs = [Path(arguments.annotation_output), Path(arguments.system_output)]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite annotation-pool output")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(pool, indent=2) + "\n", encoding="utf-8")
    if outputs[1].suffix.lower() == ".jsonl":
        with outputs[1].open("w", encoding="utf-8") as file:
            for record in sidecar["system_records"]:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        outputs[1].write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"task": arguments.task, "record_count": len(pool["records"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
