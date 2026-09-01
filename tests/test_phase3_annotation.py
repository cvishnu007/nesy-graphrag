import json
from pathlib import Path

import pytest

from src.evaluation.annotation_pool import generate_annotation_pool
from src.evaluation.phase3_annotation import (
    HYPOTHESIS_DIMENSIONS,
    POOL_FILES,
    Phase3AnnotationError,
    analyze_responses,
    finalize_phase3,
    load_completed_packets,
    prepare_annotation_workflow,
    validate_reviewer_packet,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fixture_pools(root: Path):
    for split in ("dev", "test"):
        contradiction = []
        support = []
        hypothesis = []
        for index in range(2):
            suffix = f"{split}-{index}"
            contradiction.append({
                "split": split,
                "paper1": {"id": f"a-{suffix}", "title": "A", "abstract": "Abstract A"},
                "paper2": {"id": f"b-{suffix}", "title": "B", "abstract": "Abstract B"},
                "prediction": "CONTRADICTION", "confidence": 0.9,
            })
            support.append({
                "split": split, "query_id": f"Q-{suffix}", "claim": f"Claim {suffix}",
                "passage_id": f"P-{suffix}", "passage_text": "Passage", "paper_id": f"a-{suffix}",
                "prediction": "SUPPORTED", "confidence": 0.8,
            })
            hypothesis.append({
                "split": split, "query_id": f"Q-{suffix}",
                "hypothesis": f"Hypothesis {suffix}", "evidence": [{"paper_id": f"a-{suffix}"}],
                "accepted": True, "hns": 0.7,
            })
        for task, candidates in (
            ("contradiction", contradiction), ("support", support), ("hypothesis", hypothesis)
        ):
            pool, _ = generate_annotation_pool(task, candidates)
            _write_json(root / POOL_FILES[(task, split)], pool)


def _prepare(tmp_path, name="phase3"):
    pool_dir = tmp_path / "pools"
    if not pool_dir.exists():
        _fixture_pools(pool_dir)
    output = tmp_path / name
    manifest = prepare_annotation_workflow(
        pool_dir, output, seed="fixed-seed", double_fraction=0.25,
    )
    return pool_dir, output, manifest


def _complete_packets(packet_root: Path):
    for path in packet_root.glob("reviewer_*/*.json"):
        packet = json.loads(path.read_text(encoding="utf-8"))
        for record in packet["records"]:
            if packet["task"] == "contradiction":
                response = {"label": "AGREEMENT", "reason": "Compatible", "timestamp": "2026-08-31T12:00:00Z"}
            elif packet["task"] == "support":
                response = {"label": "SUPPORTED", "notes": "Direct", "timestamp": "2026-08-31T12:00:00Z"}
            else:
                response = {dimension: 3 for dimension in HYPOTHESIS_DIMENSIONS}
                response.update({"notes": "Adequate", "timestamp": "2026-08-31T12:00:00Z"})
            record["response"] = response
        packet["status"] = "complete"
        _write_json(path, packet)


def test_prepare_is_deterministic_and_uses_reviewer_slots(tmp_path):
    _, first_dir, first = _prepare(tmp_path, "first")
    _, second_dir, second = _prepare(tmp_path, "second")
    assert first["assignments"] == second["assignments"]
    assert [slot["reviewer_id"] for slot in first["reviewer_slots"]] == [
        "reviewer_01", "reviewer_02", "reviewer_03"
    ]
    assert all(slot["assigned_person"] is None for slot in first["reviewer_slots"])
    assert all(details["double_annotated"] == 1 for details in first["task_counts"].values())
    assert (first_dir / "reviewer_packets" / "reviewer_01").is_dir()
    assert (second_dir / "reviewer_packets" / "reviewer_02").is_dir()


def test_reviewer_packets_are_blinded_and_isolated(tmp_path):
    _, output, manifest = _prepare(tmp_path)
    double_item = next(item for item in manifest["assignments"] if item["double_annotation"])
    task, split, item_id = double_item["task"], double_item["split"], double_item["item_id"]
    first_path = output / "reviewer_packets" / "reviewer_01" / f"{task}_{split}.json"
    second_path = output / "reviewer_packets" / "reviewer_02" / f"{task}_{split}.json"
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    id_field = {"contradiction": "pair_id", "support": "item_id", "hypothesis": "hypothesis_id"}[task]
    first_record = next(record for record in first["records"] if record[id_field] == item_id)
    second_record = next(record for record in second["records"] if record[id_field] == item_id)
    first_record["response"] = {"private": "reviewer one value"}
    assert second_record["response"] is None
    assert "prediction" not in json.dumps(first)
    validate_reviewer_packet(second, require_complete=False)


def test_packet_rejects_non_anonymized_reviewer_and_missing_response(tmp_path):
    _, output, _ = _prepare(tmp_path)
    path = next((output / "reviewer_packets" / "reviewer_01").glob("*.json"))
    packet = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(Phase3AnnotationError, match="no response"):
        validate_reviewer_packet(packet, require_complete=True)
    packet["reviewer_id"] = "Alice Example"
    with pytest.raises(Phase3AnnotationError, match="anonymized"):
        validate_reviewer_packet(packet, require_complete=False)


def test_human_only_agreement_and_disagreement_queue(tmp_path):
    _, output, manifest = _prepare(tmp_path)
    _complete_packets(output / "reviewer_packets")
    judgments = load_completed_packets(output / "reviewer_packets")
    double_support = next(
        item for item in manifest["assignments"]
        if item["task"] == "support" and item["double_annotation"]
    )
    judgments[double_support["item_id"]]["reviewer_02"]["label"] = "UNSUPPORTED"
    agreement, queue = analyze_responses(manifest, judgments)
    assert agreement["tasks"]["support"] == {
        "double_annotated": 1, "agreeing": 0, "disagreeing": 1,
        "agreement_rate": 0.0, "cohen_kappa": 0.0,
    }
    assert queue["records"][0]["item_id"] == double_support["item_id"]
    assert "prediction" not in json.dumps(queue)


def test_finalization_reuses_existing_draft_finalizer_and_preserves_originals(tmp_path):
    pool_dir, output, manifest = _prepare(tmp_path)
    _complete_packets(output / "reviewer_packets")
    judgments = load_completed_packets(output / "reviewer_packets")
    double_pair = next(
        item for item in manifest["assignments"]
        if item["task"] == "contradiction" and item["double_annotation"]
    )
    item_id = double_pair["item_id"]
    judgments[item_id]["reviewer_02"]["label"] = "DIFFERENT SCOPE"
    # Persist the deliberately independent disagreement to reviewer 2's packet.
    path = output / "reviewer_packets" / "reviewer_02" / f"contradiction_{double_pair['split']}.json"
    packet = json.loads(path.read_text(encoding="utf-8"))
    record = next(record for record in packet["records"] if record["pair_id"] == item_id)
    record["response"]["label"] = "DIFFERENT SCOPE"
    record["response"]["reason"] = "Different conditions"
    _write_json(path, packet)
    adjudications = {
        "version": "1.0", "records": [{
            "item_id": item_id, "adjudicator_id": "reviewer_03",
            "response": {"label": "AGREEMENT", "reason": "Adjudicated from abstracts", "timestamp": "2026-08-31T13:00:00Z"},
        }],
    }
    adjudication_path = output / "adjudications.json"
    _write_json(adjudication_path, adjudications)
    benchmarks = finalize_phase3(
        pool_dir, output / "assignment_manifest.json", output / "reviewer_packets",
        adjudication_path, output / "annotated_pools", output / "benchmarks",
    )
    assert all(benchmark["status"] == "draft" for benchmark in benchmarks.values())
    annotated_path = output / "annotated_pools" / f"contradiction_{double_pair['split']}.json"
    annotated = json.loads(annotated_path.read_text(encoding="utf-8"))
    final_record = next(record for record in annotated["records"] if record["pair_id"] == item_id)
    assert [item["label"] for item in final_record["annotations"]] == ["AGREEMENT", "DIFFERENT SCOPE"]
    assert final_record["adjudication"]["label"] == "AGREEMENT"
    assert final_record["adjudication"]["reviewer_id"] == "reviewer_03"
