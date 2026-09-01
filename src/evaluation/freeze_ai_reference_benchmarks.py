"""Validate and freeze the completed AI-reference reasoning benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from src.evaluation.reasoning_benchmark_io import load_reasoning_benchmark


FILES = {
    "contradiction": "contradiction_pairs.json",
    "support": "claim_support.json",
    "hypothesis": "hypothesis_ratings.json",
}
COLLECTIONS = {
    "contradiction": "pairs",
    "support": "items",
    "hypothesis": "hypotheses",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def freeze_ai_reference_benchmark(
    benchmark: Mapping,
    *,
    task: str,
    source_sha256: str,
    frozen_at_utc: str | None = None,
) -> dict:
    if task not in COLLECTIONS:
        raise ValueError(f"Unknown task: {task}")
    if benchmark.get("status") != "draft":
        raise ValueError("Only a draft benchmark can be frozen")
    if benchmark.get("benchmark_version") != "1.0-ai-reference-draft":
        raise ValueError("Unexpected source benchmark version")
    provenance = benchmark.get("annotation_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("AI-reference annotation provenance is required")
    if provenance.get("human_ground_truth") is not False:
        raise ValueError("Benchmark must explicitly deny human ground truth")
    source = str(provenance.get("annotation_source", ""))
    if "AI-generated" not in source:
        raise ValueError("Benchmark must declare AI-generated references")
    if len(source_sha256) != 64:
        raise ValueError("source_sha256 must be a SHA-256 hex digest")

    frozen = json.loads(json.dumps(benchmark))
    frozen["status"] = "frozen"
    frozen["benchmark_version"] = "1.0-ai-reference-frozen"
    frozen["freeze_metadata"] = {
        "frozen_at_utc": frozen_at_utc or _now(),
        "source_benchmark_version": benchmark["benchmark_version"],
        "source_sha256": source_sha256,
        "labels_changed_during_freeze": False,
        "reference_annotation_source": "AI-generated",
        "human_ground_truth": False,
    }
    return frozen


def freeze_benchmarks(
    paths: Mapping[str, Path],
    manifest_path: Path,
    *,
    frozen_at_utc: str | None = None,
) -> dict:
    timestamp = frozen_at_utc or _now()
    entries = {}
    frozen_values = {}
    for task, raw_path in paths.items():
        path = Path(raw_path)
        source_bytes = path.read_bytes()
        source_hash = _sha256(source_bytes)
        benchmark = load_reasoning_benchmark(path, task)
        frozen = freeze_ai_reference_benchmark(
            benchmark,
            task=task,
            source_sha256=source_hash,
            frozen_at_utc=timestamp,
        )
        collection = COLLECTIONS[task]
        split_counts = {
            split: sum(item["split"] == split for item in frozen[collection])
            for split in ("dev", "test")
        }
        frozen_values[task] = (path, frozen)
        entries[task] = {
            "path": path.as_posix(),
            "benchmark_version": "1.0-ai-reference-frozen",
            "status": "frozen",
            "count": len(frozen[collection]),
            "split_counts": split_counts,
            "source_sha256": source_hash,
            "sha256": "",
            "reference_annotation_source": "AI-generated",
            "human_ground_truth": False,
        }

    for task, (path, frozen) in frozen_values.items():
        _write_json_atomic(path, frozen)
        load_reasoning_benchmark(path, task)
        entries[task]["sha256"] = _sha256(path.read_bytes())

    manifest = {
        "manifest_version": "1.0",
        "status": "complete",
        "frozen_at_utc": timestamp,
        "reference_annotation_source": "AI-generated",
        "human_ground_truth": False,
        "benchmarks": entries,
    }
    _write_json_atomic(Path(manifest_path), manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default="evaluation/benchmarks")
    parser.add_argument(
        "--manifest",
        default="evaluation/benchmarks/ai_reference_frozen_manifest.json",
    )
    args = parser.parse_args(argv)
    directory = Path(args.benchmark_dir)
    manifest = freeze_benchmarks(
        {task: directory / filename for task, filename in FILES.items()},
        Path(args.manifest),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
