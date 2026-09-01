import json
import sys
from types import SimpleNamespace

from src.evaluation.phase3_ai_annotation import (
    AI_SOURCE,
    _collect_annotation_models,
    _groq_judge,
    _judge_validated,
    _parse_judge_output,
    annotate_packets,
    find_ai_disagreements,
)


def _packet(task="support"):
    if task == "support":
        records = [{
            "item_id": "S1", "split": "dev", "query_id": "Q1",
            "claim": "Claim", "passage_id": "P1", "passage_text": "Evidence",
            "paper_id": "paper-1", "response": None,
        }, {
            "item_id": "S2", "split": "dev", "query_id": "Q1",
            "claim": "Other", "passage_id": "P2", "passage_text": "Other evidence",
            "paper_id": "paper-2", "response": {
                "label": "UNSUPPORTED", "notes": "existing",
                "timestamp": "2026-08-31T00:00:00Z",
            },
        }]
    else:
        records = []
    return {
        "packet_version": "1.0", "status": "unstarted", "blinded": True,
        "reviewer_id": "reviewer_01", "task": task, "split": "dev",
        "response_schema": {}, "records": records,
    }


def test_ai_output_is_filtered_validated_and_marked_non_human():
    raw = {"annotations": [{
        "id": "S1", "response": {
            "label": "SUPPORTED", "notes": "Direct", "confidence": 0.99,
        },
    }]}
    result = _parse_judge_output(raw, "support", ["S1"], "fixture-ai", "reviewer_01")["S1"]
    assert result["label"] == "SUPPORTED"
    assert "confidence" not in result
    assert result["annotation_source"] == AI_SOURCE
    assert result["independent_human_annotation"] is False


def test_resumable_annotation_preserves_existing_response(tmp_path):
    packet_dir = tmp_path / "packets" / "reviewer_01"
    packet_dir.mkdir(parents=True)
    path = packet_dir / "support_dev.json"
    packet = _packet()
    existing = dict(packet["records"][1]["response"])
    path.write_text(json.dumps(packet), encoding="utf-8")

    def judge(_prompt):
        return {"annotations": [{
            "id": "S1", "response": {"label": "SUPPORTED", "notes": "visible evidence"},
        }]}

    summary = annotate_packets(tmp_path / "packets", judge, model="fixture-ai")
    completed = json.loads(path.read_text(encoding="utf-8"))
    assert summary["generated"] == 1 and summary["preserved"] == 1
    assert completed["records"][1]["response"] == existing
    assert completed["records"][0]["response"]["annotation_source"] == AI_SOURCE
    assert completed["annotation_methodology"]["human_ground_truth"] is False


def test_disagreements_are_ai_pass_differences_not_human_agreement():
    manifest = {"assignments": [{
        "item_id": "S1", "task": "support", "split": "dev",
        "reviewer_ids": ["reviewer_01", "reviewer_02"], "double_annotation": True,
    }]}
    judgments = {"S1": {
        "reviewer_01": {"label": "SUPPORTED"},
        "reviewer_02": {"label": "PARTIALLY_SUPPORTED"},
    }}
    result = find_ai_disagreements(manifest, judgments)
    assert result[0]["item_id"] == "S1"
    assert {row["slot"] for row in result[0]["responses"]} == {"reviewer_01", "reviewer_02"}


def test_groq_judge_reserves_output_and_retries_across_tpm_window(monkeypatch):
    captured = {}

    class FakeGroq:
        def __init__(self, *, api_key):
            captured["api_key"] = api_key

    def fake_chat(_client, prompt, **kwargs):
        captured.update(prompt=prompt, **kwargs)
        return {"annotations": []}

    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    monkeypatch.setattr("src.utils.config.GROQ_API_KEY", "fixture-key")
    monkeypatch.setattr("src.utils.config.is_configured", lambda value: value == "fixture-key")
    monkeypatch.setattr("src.utils.groq_client.groq_chat_with_retry", fake_chat)

    result = _groq_judge("fixture-model")("fixture prompt")

    assert result == {"annotations": []}
    assert captured["max_tokens"] == 2000
    assert captured["max_retries"] == 6


def test_default_annotation_batches_stay_within_prompt_budget(tmp_path):
    packet_dir = tmp_path / "packets" / "reviewer_01"
    packet_dir.mkdir(parents=True)
    path = packet_dir / "support_dev.json"
    packet = _packet()
    packet["records"] = [{
        "item_id": f"S{index}", "split": "dev", "query_id": "Q1",
        "claim": f"Claim {index}", "passage_id": f"P{index}",
        "passage_text": "Visible evidence. " * 80,
        "paper_id": f"paper-{index}", "response": None,
    } for index in range(12)]
    path.write_text(json.dumps(packet), encoding="utf-8")

    prompt_lengths = []

    def judge(prompt):
        prompt_lengths.append(len(prompt))
        assert len(prompt) <= 8000
        visible = json.loads(prompt.split("VISIBLE RECORDS:\n", 1)[1])
        return {"annotations": [{
            "id": record["item_id"],
            "response": {"label": "SUPPORTED", "notes": "Visible evidence"},
        } for record in visible]}

    summary = annotate_packets(tmp_path / "packets", judge, model="fixture-ai")

    assert summary["generated"] == 12
    assert len(prompt_lengths) > 1


def test_default_annotation_retries_transient_malformed_json(tmp_path):
    packet_dir = tmp_path / "packets" / "reviewer_01"
    packet_dir.mkdir(parents=True)
    path = packet_dir / "support_dev.json"
    packet = _packet()
    packet["records"] = packet["records"][:1]
    path.write_text(json.dumps(packet), encoding="utf-8")
    attempts = 0

    def judge(_prompt):
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            return "{malformed"
        return {"annotations": [{
            "id": "S1",
            "response": {"label": "SUPPORTED", "notes": "Visible evidence"},
        }]}

    summary = annotate_packets(tmp_path / "packets", judge, model="fixture-ai")

    assert attempts == 4
    assert summary["generated"] == 1


def test_packet_methodology_lists_preserved_and_new_ai_models(tmp_path):
    packet_dir = tmp_path / "packets" / "reviewer_01"
    packet_dir.mkdir(parents=True)
    path = packet_dir / "support_dev.json"
    packet = _packet()
    packet["records"][1]["response"].update({
        "annotation_source": AI_SOURCE,
        "annotation_model": "model-a",
        "independent_human_annotation": False,
    })
    path.write_text(json.dumps(packet), encoding="utf-8")

    def judge(_prompt):
        return {"annotations": [{
            "id": "S1",
            "response": {"label": "SUPPORTED", "notes": "Visible evidence"},
        }]}

    annotate_packets(tmp_path / "packets", judge, model="model-b")
    completed = json.loads(path.read_text(encoding="utf-8"))

    assert completed["annotation_methodology"]["models"] == ["model-a", "model-b"]
    assert "model" not in completed["annotation_methodology"]


def test_final_provenance_collects_pass_and_consensus_models():
    judgments = {
        "S1": {"reviewer_01": {"annotation_model": "model-a"}},
        "S2": {"reviewer_02": {"annotation_model": "model-b"}},
        "legacy": {"reviewer_01": {"label": "SUPPORTED"}},
    }
    consensus = {"records": [{"response": {"annotation_model": "model-c"}}]}

    assert _collect_annotation_models(judgments, consensus) == [
        "model-a", "model-b", "model-c",
    ]


def test_default_support_batch_size_avoids_large_json_responses(tmp_path):
    packet_dir = tmp_path / "packets" / "reviewer_01"
    packet_dir.mkdir(parents=True)
    path = packet_dir / "support_dev.json"
    packet = _packet()
    packet["records"] = [{
        "item_id": f"S{index}", "split": "dev", "query_id": "Q1",
        "claim": f"Claim {index}", "passage_id": f"P{index}",
        "passage_text": "Evidence", "paper_id": f"paper-{index}",
        "response": None,
    } for index in range(12)]
    path.write_text(json.dumps(packet), encoding="utf-8")
    batch_sizes = []

    def judge(prompt):
        visible = json.loads(prompt.split("VISIBLE RECORDS:\n", 1)[1])
        batch_sizes.append(len(visible))
        if len(visible) > 6:
            return "{malformed"
        return {"annotations": [{
            "id": record["item_id"],
            "response": {"label": "SUPPORTED", "notes": "Visible evidence"},
        } for record in visible]}

    summary = annotate_packets(tmp_path / "packets", judge, model="fixture-ai")

    assert summary["generated"] == 12
    assert batch_sizes == [6, 6]


def test_parse_retry_includes_validation_error_for_correction(tmp_path):
    packet_dir = tmp_path / "packets" / "reviewer_01"
    packet_dir.mkdir(parents=True)
    path = packet_dir / "hypothesis_dev.json"
    packet = {
        "packet_version": "1.0", "status": "unstarted", "blinded": True,
        "reviewer_id": "reviewer_01", "task": "hypothesis", "split": "dev",
        "response_schema": {}, "records": [{
            "hypothesis_id": "H1", "split": "dev", "query_id": "Q1",
            "hypothesis": "A specific testable hypothesis", "evidence": [],
            "response": None,
        }],
    }
    path.write_text(json.dumps(packet), encoding="utf-8")
    prompts = []

    def judge(prompt):
        prompts.append(prompt)
        specificity = 3 if "specificity must be 1, 3, or 5" in prompt else 4
        return {"annotations": [{
            "id": "H1", "response": {
                "evidence": 3, "novelty": 3, "feasibility": 3,
                "specificity": specificity, "usefulness": 3, "notes": "Visible evidence",
            },
        }]}

    summary = annotate_packets(tmp_path / "packets", judge, model="fixture-ai")

    assert summary["generated"] == 1
    assert len(prompts) == 2


def test_consensus_validation_helper_corrects_invalid_scores():
    prompts = []

    def judge(prompt):
        prompts.append(prompt)
        score = 3 if "evidence must be 1, 3, or 5" in prompt else 2
        return {"annotations": [{
            "id": "H1", "response": {
                "evidence": score, "novelty": 3, "feasibility": 3,
                "specificity": 3, "usefulness": 3, "notes": "Visible evidence",
            },
        }]}

    parsed = _judge_validated(
        judge, "original prompt", "hypothesis", ["H1"],
        "fixture-ai", "reviewer_03",
    )

    assert parsed["H1"]["evidence"] == 3
    assert len(prompts) == 2
