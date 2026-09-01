"""Controlled fixed-sample comparison of two concept extractors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Callable, Iterable

from src.ingestion.ner_extractor import extract_entities, filter_entities


SCIENTIFIC_TERMS = (
    "graph neural network", "graph convolutional network", "graph attention network",
    "message passing", "node classification", "link prediction", "graph classification",
    "collaborative filtering", "recommendation system", "knowledge graph",
    "neural network", "deep learning", "machine learning", "attention mechanism",
    "representation learning", "contrastive learning", "self-supervised learning",
    "natural language processing", "computer vision", "molecular graph",
    "oversmoothing", "over-squashing", "heterophily", "homophily",
    "precision", "recall", "f1 score", "mean reciprocal rank", "ndcg",
)
MODEL_TOKEN = re.compile(
    r"\b(?:[A-Za-z0-9-]*(?:GNN|GCN)[A-Za-z0-9-]*|GATs?|GraphSAGE|BERT|Transformers?)\b",
    re.IGNORECASE,
)


def extract_scientific_pattern_concepts(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    lowered = text.casefold()
    concepts = {
        term for term in SCIENTIFIC_TERMS
        if re.search(rf"\b{re.escape(term)}s?\b", lowered)
    }
    concepts.update(match.group(0).casefold() for match in MODEL_TOKEN.finditer(text))
    return sorted(filter_entities(concepts))


def _sample(records: Iterable[dict], sample_size: int, seed: str) -> list[dict]:
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    rows = list(records)
    if any(not isinstance(row.get("id"), str) or not row["id"] for row in rows):
        raise ValueError("Every corpus record requires a non-empty id")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Corpus IDs must be unique")
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}\x1f{row['id']}".encode("utf-8")
        ).hexdigest(),
    )[:sample_size]


def _summary(outputs: list[dict], elapsed_seconds: float) -> dict:
    counts = [len(row["concepts"]) for row in outputs]
    frequencies = Counter(concept for row in outputs for concept in row["concepts"])
    return {
        "document_ids": [row["document_id"] for row in outputs],
        "document_count": len(outputs),
        "total_concepts": sum(counts),
        "unique_concepts": len(frequencies),
        "concepts_per_document_mean": fmean(counts) if counts else 0.0,
        "empty_document_rate": (
            sum(count == 0 for count in counts) / len(counts) if counts else 0.0
        ),
        "runtime_seconds": elapsed_seconds,
        "top_concepts": [
            {"concept": concept, "count": count}
            for concept, count in frequencies.most_common(25)
        ],
    }


def compare_ner_extractors(
    records: Iterable[dict],
    baseline_extractor: Callable[[str], list[str]],
    alternative_extractor: Callable[[str], list[str]],
    *,
    sample_size: int,
    seed: str,
) -> dict:
    sample = _sample(records, sample_size, seed)
    outputs = {}
    summaries = {}
    for name, extractor in (
        ("baseline", baseline_extractor), ("alternative", alternative_extractor)
    ):
        started = perf_counter()
        rows = []
        for record in sample:
            text = str(record.get("clean_abstract") or record.get("abstract") or "")
            concepts = sorted(set(filter_entities(extractor(text))))
            rows.append({"document_id": record["id"], "concepts": concepts})
        elapsed = perf_counter() - started
        outputs[name] = rows
        summaries[name] = _summary(rows, elapsed)
    per_document_jaccard = []
    for baseline, alternative in zip(outputs["baseline"], outputs["alternative"]):
        left, right = set(baseline["concepts"]), set(alternative["concepts"])
        union = left | right
        per_document_jaccard.append(len(left & right) / len(union) if union else 1.0)
    return {
        "sample_size_requested": sample_size,
        "sample_size_actual": len(sample),
        "seed": seed,
        "document_ids": [row["id"] for row in sample],
        "baseline": summaries["baseline"],
        "alternative": summaries["alternative"],
        "overlap": {
            "mean_document_jaccard": fmean(per_document_jaccard) if per_document_jaccard else 0.0,
        },
        "outputs": outputs,
        "controlled_variables": {
            "only_changed_component": "concept_extractor",
            "identical_documents": True,
            "identical_text_field": True,
            "production_graph_modified": False,
        },
    }


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
    parser.add_argument("--input", required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", default="ner-comparison-v1")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    import spacy

    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    nlp = spacy.load("en_core_web_sm")
    baseline = lambda text: filter_entities(extract_entities(nlp, text))
    result = compare_ner_extractors(
        records, baseline, extract_scientific_pattern_concepts,
        sample_size=args.sample_size, seed=args.seed,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_outputs = result.pop("outputs")
    (output / "comparison_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, rows in raw_outputs.items():
        with (output / f"{name}_concepts.jsonl").open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "failures.jsonl").write_text("", encoding="utf-8")
    command = (
        ".\\venv\\Scripts\\python.exe -m src.evaluation.ner_comparison "
        f"--input {args.input} --sample-size {args.sample_size} --seed {args.seed} "
        f"--output-dir {args.output_dir}"
    )
    metadata = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input": args.input,
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "baseline": "en_core_web_sm entities plus noun chunks",
        "baseline_version": importlib.metadata.version("en-core-web-sm"),
        "alternative": "fixed scientific term and model-token patterns",
        "sample_size": result["sample_size_actual"],
        "seed": args.seed,
        "failure_count": 0,
        "reproduction_command": command,
        "controlled_variables": result["controlled_variables"],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
