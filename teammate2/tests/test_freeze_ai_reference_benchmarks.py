import copy
import hashlib
import json

from src.evaluation.freeze_ai_reference_benchmarks import (
    freeze_ai_reference_benchmark,
    freeze_benchmarks,
)


def support_benchmark():
    return {
        "benchmark_version": "1.0-ai-reference-draft",
        "status": "draft",
        "fixture_only": False,
        "items": [
            {
                "item_id": "S1", "split": "dev", "query_id": "Q1",
                "claim": "claim", "passage_id": "P1",
                "passage_text": "passage", "paper_id": "paper-1",
                "label": "SUPPORTED", "notes": "AI note",
            },
            {
                "item_id": "S2", "split": "test", "query_id": "Q1",
                "claim": "claim 2", "passage_id": "P2",
                "passage_text": "passage 2", "paper_id": "paper-2",
                "label": "UNSUPPORTED", "notes": "AI note",
            },
        ],
        "annotation_provenance": {
            "annotation_source": "AI-generated",
            "annotation_models": ["model-a"],
            "human_ground_truth": False,
            "independent_human_review": False,
        },
    }


def test_freeze_changes_only_status_version_and_adds_metadata():
    before = support_benchmark()
    original = copy.deepcopy(before)

    frozen = freeze_ai_reference_benchmark(
        before, task="support", source_sha256="a" * 64,
        frozen_at_utc="2026-09-01T00:00:00Z",
    )

    assert before == original
    assert frozen["items"] == original["items"]
    assert frozen["status"] == "frozen"
    assert frozen["benchmark_version"] == "1.0-ai-reference-frozen"
    assert frozen["freeze_metadata"] == {
        "frozen_at_utc": "2026-09-01T00:00:00Z",
        "source_benchmark_version": "1.0-ai-reference-draft",
        "source_sha256": "a" * 64,
        "labels_changed_during_freeze": False,
        "reference_annotation_source": "AI-generated",
        "human_ground_truth": False,
    }


def test_freeze_benchmarks_writes_hashes_counts_and_splits(tmp_path):
    benchmark_dir = tmp_path / "benchmarks"
    benchmark_dir.mkdir()
    path = benchmark_dir / "claim_support.json"
    path.write_text(json.dumps(support_benchmark()), encoding="utf-8")
    manifest_path = benchmark_dir / "manifest.json"

    manifest = freeze_benchmarks(
        {"support": path}, manifest_path,
        frozen_at_utc="2026-09-01T00:00:00Z",
    )

    frozen_bytes = path.read_bytes()
    entry = manifest["benchmarks"]["support"]
    assert entry["count"] == 2
    assert entry["split_counts"] == {"dev": 1, "test": 1}
    assert entry["sha256"] == hashlib.sha256(frozen_bytes).hexdigest()
    assert len(entry["source_sha256"]) == 64
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
