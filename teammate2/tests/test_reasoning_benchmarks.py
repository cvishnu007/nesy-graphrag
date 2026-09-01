import json

import pytest

from src.evaluation.reasoning_benchmark_io import (
    ReasoningBenchmarkValidationError,
    load_reasoning_benchmark,
    records_for_split,
    validate_claim_support_benchmark,
    validate_contradiction_benchmark,
    validate_hypothesis_benchmark,
)
from src.evaluation.reasoning_runner import build_parser, run_reasoning_evaluation


def contradiction_benchmark():
    return {
        "benchmark_version": "1.0", "status": "frozen",
        "pairs": [{
            "pair_id": "C001", "split": "dev", "paper1_id": "a",
            "paper2_id": "b", "label": "DIFFERENT SCOPE", "reason": "different task",
            "annotators": ["A1", "A2"], "adjudicated": True,
        }],
    }


def support_benchmark():
    return {
        "benchmark_version": "1.0", "status": "frozen",
        "items": [{
            "item_id": "S001", "split": "test", "query_id": "Q001",
            "claim": "Claim", "passage_id": "P1-S001", "passage_text": "Evidence",
            "paper_id": "p1", "label": "SUPPORTED", "notes": "direct",
        }],
    }


def hypothesis_benchmark():
    return {
        "benchmark_version": "1.0", "status": "frozen",
        "hypotheses": [{
            "hypothesis_id": "H001", "split": "dev", "query_id": "Q001",
            "hypothesis": "Combine A and B", "hns": 0.5,
            "ratings": [{
                "reviewer_id": "R1", "evidence": 3, "novelty": 5,
                "feasibility": 3, "specificity": 3, "usefulness": 5,
            }],
        }],
    }


def test_valid_reasoning_benchmarks():
    validate_contradiction_benchmark(contradiction_benchmark(), valid_paper_ids={"a", "b"})
    validate_claim_support_benchmark(
        support_benchmark(), valid_query_ids={"Q001"}, valid_paper_ids={"p1"},
        valid_passage_ids={"P1-S001"},
    )
    validate_hypothesis_benchmark(hypothesis_benchmark(), valid_query_ids={"Q001"})


def test_empty_draft_containers_are_valid():
    validate_contradiction_benchmark({"benchmark_version": "1.0-draft", "status": "draft", "pairs": []})
    validate_claim_support_benchmark({"benchmark_version": "1.0-draft", "status": "draft", "items": []})
    validate_hypothesis_benchmark({"benchmark_version": "1.0-draft", "status": "draft", "hypotheses": []})


def test_reversed_and_duplicate_pairs_are_rejected():
    benchmark = contradiction_benchmark()
    benchmark["pairs"].append({**benchmark["pairs"][0], "pair_id": "C002", "paper1_id": "b", "paper2_id": "a"})
    with pytest.raises(ReasoningBenchmarkValidationError, match="canonical"):
        validate_contradiction_benchmark(benchmark)
    benchmark["pairs"][1].update({"paper1_id": "a", "paper2_id": "b"})
    with pytest.raises(ReasoningBenchmarkValidationError, match="Duplicate or reversed"):
        validate_contradiction_benchmark(benchmark)


@pytest.mark.parametrize("label", ["UNKNOWN", "CONTRADICTED", ""])
def test_invalid_contradiction_gold_label(label):
    benchmark = contradiction_benchmark()
    benchmark["pairs"][0]["label"] = label
    with pytest.raises(ReasoningBenchmarkValidationError):
        validate_contradiction_benchmark(benchmark)


def test_invalid_support_reference_and_duplicate_item():
    benchmark = support_benchmark()
    with pytest.raises(ReasoningBenchmarkValidationError, match="Unknown reference"):
        validate_claim_support_benchmark(benchmark, valid_passage_ids={"other"})
    benchmark["items"].append(dict(benchmark["items"][0]))
    with pytest.raises(ReasoningBenchmarkValidationError, match="Duplicate item_id"):
        validate_claim_support_benchmark(benchmark)


@pytest.mark.parametrize("score", [0, 2, 4, 6, True])
def test_invalid_hypothesis_scores(score):
    benchmark = hypothesis_benchmark()
    benchmark["hypotheses"][0]["ratings"][0]["novelty"] = score
    with pytest.raises(ReasoningBenchmarkValidationError):
        validate_hypothesis_benchmark(benchmark)


def test_loader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"benchmark_version":"1","benchmark_version":"2"}', encoding="utf-8")
    with pytest.raises(ReasoningBenchmarkValidationError, match="Duplicate JSON key"):
        load_reasoning_benchmark(path, "support")


def test_split_filter_and_invalid_split():
    assert records_for_split(contradiction_benchmark(), "contradiction", "dev")[0]["pair_id"] == "C001"
    with pytest.raises(ReasoningBenchmarkValidationError):
        records_for_split(contradiction_benchmark(), "contradiction", "all")


def test_runner_validates_arguments():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--tasks", "bad", "--split", "dev"])
    with pytest.raises(ValueError):
        run_reasoning_evaluation(tasks=[], split="dev", output_dir="unused")


def test_runner_handles_empty_offline_benchmarks(tmp_path):
    paths = {}
    for task, collection in (("contradiction", "pairs"), ("support", "items"), ("hypothesis", "hypotheses")):
        path = tmp_path / f"{task}.json"
        path.write_text(json.dumps({"benchmark_version": "1.0-draft", "status": "draft", collection: []}), encoding="utf-8")
        paths[task] = path
    output = tmp_path / "output"
    result = run_reasoning_evaluation(
        tasks=["contradiction", "support", "hypothesis"], split="dev",
        output_dir=output, benchmark_paths=paths,
    )
    assert len(result["failures"]) == 3
    assert {item["code"] for item in result["failures"]} == {"no_benchmark_data"}
    assert (output / "metadata.json").exists()
    assert (output / "failures.jsonl").exists()
    with pytest.raises(FileExistsError):
        run_reasoning_evaluation(tasks=["support"], split="dev", output_dir=output, benchmark_paths=paths)
