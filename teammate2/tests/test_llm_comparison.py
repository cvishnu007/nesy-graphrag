import json

from src.evaluation.llm_comparison import (
    build_llm_comparison_prompt,
    run_llm_comparison,
)


def record(pair_id="C1", label="AGREEMENT"):
    return {
        "pair_id": pair_id,
        "split": "dev",
        "paper1_id": "A",
        "paper2_id": "B",
        "paper1_title": "First",
        "paper1_abstract": "First abstract",
        "paper2_title": "Second",
        "paper2_abstract": "Second abstract",
        "label": label,
        "reason": "hidden reference note",
        "annotators": ["ai-pass"],
        "adjudicated": False,
    }


def test_llm_prompt_excludes_all_reference_fields():
    prompt = build_llm_comparison_prompt([record()])

    assert "First abstract" in prompt
    assert '"label"' not in prompt
    assert "hidden reference note" not in prompt
    assert "annotators" not in prompt
    assert "adjudicated" not in prompt


def test_comparison_changes_only_model_and_resumes_checkpoints(tmp_path):
    calls = {"first": 0, "second": 0}

    def judge(name):
        def run(prompt):
            calls[name] += 1
            return json.dumps({
                "annotations": [{
                    "id": "C1", "prediction": "AGREEMENT",
                    "confidence": 0.9, "reason": "compatible claims",
                }]
            })
        return run

    judges = {"first": judge("first"), "second": judge("second")}
    first = run_llm_comparison(
        [record()], judges, {"first": "model-a", "second": "model-b"},
        output_dir=tmp_path, split="dev", batch_size=4, threshold=0.5,
    )
    second = run_llm_comparison(
        [record()], judges, {"first": "model-a", "second": "model-b"},
        output_dir=tmp_path, split="dev", batch_size=4, threshold=0.5,
    )

    assert calls == {"first": 1, "second": 1}
    assert first["controlled_variables"]["only_changed_component"] == "llm_model"
    assert first["models"]["first"]["prompt_hashes"] == first["models"]["second"]["prompt_hashes"]
    assert second["models"]["first"]["metrics"]["accuracy"] == 1.0


def test_invalid_response_is_retried_and_audited(tmp_path):
    calls = {"first": 0, "second": 0}

    def judge(name):
        def run(prompt):
            calls[name] += 1
            if name == "first" and calls[name] == 1:
                return ""
            return json.dumps({
                "annotations": [{
                    "id": "C1", "prediction": "AGREEMENT",
                    "confidence": 0.9, "reason": "compatible claims",
                }]
            })
        return run

    result = run_llm_comparison(
        [record()], {name: judge(name) for name in calls},
        {"first": "model-a", "second": "model-b"},
        output_dir=tmp_path, split="dev", batch_size=4, threshold=0.5,
    )

    assert calls == {"first": 2, "second": 1}
    assert result["models"]["first"]["invalid_response_count"] == 1
    failures = (tmp_path / "failures.jsonl").read_text(encoding="utf-8")
    assert "JSONDecodeError" in failures
