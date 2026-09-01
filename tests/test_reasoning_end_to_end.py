import json
from pathlib import Path

from src.evaluation.reasoning_runner import _reference_metadata, run_reasoning_evaluation


FIXTURES = Path(__file__).parent / "fixtures" / "reasoning"


def test_fixture_only_end_to_end_reasoning_runner(tmp_path):
    output = tmp_path / "reasoning-output"
    result = run_reasoning_evaluation(
        tasks=["contradiction", "support", "hypothesis"],
        split="dev",
        output_dir=output,
        benchmark_paths={
            "contradiction": FIXTURES / "contradiction_benchmark.json",
            "support": FIXTURES / "claim_support_benchmark.json",
            "hypothesis": FIXTURES / "hypothesis_benchmark.json",
        },
        prediction_paths={
            "contradiction": FIXTURES / "contradiction_predictions.jsonl",
            "support": FIXTURES / "claim_support_predictions.jsonl",
        },
    )

    assert result["failures"] == []
    assert result["metrics"]["contradiction"]["status"] == "complete"
    assert result["metrics"]["support"]["status"] == "complete"
    assert result["metadata"]["test_fixture_only"] is True
    assert result["metadata"]["fixture_only_by_task"] == {
        "contradiction": True, "support": True, "hypothesis": True,
    }

    expected = {
        "metadata.json", "failures.jsonl", "contradiction_predictions.jsonl",
        "contradiction_metrics.json", "claim_support_predictions.jsonl",
        "claim_support_metrics.json", "hypothesis_ratings.csv",
        "hypothesis_metrics.json",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    saved_metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert saved_metadata["test_fixture_only"] is True
    assert (output / "failures.jsonl").read_text(encoding="utf-8") == ""


def test_reference_metadata_preserves_explicit_ai_non_human_provenance():
    result = _reference_metadata({
        "annotation_provenance": {
            "annotation_source": "AI-generated",
            "human_ground_truth": False,
            "independent_human_review": False,
        }
    })

    assert result == {
        "reference_annotation_source": "AI-generated",
        "human_ground_truth": False,
        "independent_human_review": False,
    }
