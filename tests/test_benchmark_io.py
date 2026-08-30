import json
from copy import deepcopy

import pytest

from src.evaluation.benchmark_io import (
    BenchmarkValidationError,
    load_benchmark,
    queries_for_split,
    validate_benchmark,
)


def make_valid_benchmark():
    queries = []

    for number in range(1, 21):
        queries.append(
            {
                "query_id": f"Q{number:03d}",
                "split": "dev" if number <= 6 else "test",
                "query": f"unique retrieval query {number}",
                "judgments": {
                    f"paper-{number}": 2,
                },
            }
        )

    return {
        "benchmark_version": "1.0",
        "corpus": {
            "source": "s2",
            "paper_count": 8850,
            "years": "2020-2025",
            "collection": "s2_papers",
        },
        "queries": queries,
    }


def test_valid_benchmark_is_accepted():
    validate_benchmark(
        make_valid_benchmark(),
        require_judgments=True,
    )


def test_draft_allows_empty_judgments():
    benchmark = make_valid_benchmark()

    for query in benchmark["queries"]:
        query["judgments"] = {}

    validate_benchmark(
        benchmark,
        require_judgments=False,
    )


def test_final_benchmark_rejects_empty_judgments():
    benchmark = make_valid_benchmark()
    benchmark["queries"][0]["judgments"] = {}

    with pytest.raises(BenchmarkValidationError):
        validate_benchmark(
            benchmark,
            require_judgments=True,
        )


def test_final_benchmark_requires_relevant_paper():
    benchmark = make_valid_benchmark()
    benchmark["queries"][0]["judgments"] = {
        "paper-1": 0,
    }

    with pytest.raises(BenchmarkValidationError):
        validate_benchmark(
            benchmark,
            require_judgments=True,
        )


def test_duplicate_query_id_is_rejected():
    benchmark = make_valid_benchmark()
    benchmark["queries"][1]["query_id"] = "Q001"

    with pytest.raises(BenchmarkValidationError):
        validate_benchmark(benchmark)


def test_duplicate_query_text_is_rejected():
    benchmark = make_valid_benchmark()
    benchmark["queries"][1]["query"] = (
        benchmark["queries"][0]["query"].upper()
    )

    with pytest.raises(BenchmarkValidationError):
        validate_benchmark(benchmark)


@pytest.mark.parametrize(
    "missing_field",
    ["query_id", "query", "split", "judgments"],
)
def test_missing_query_field_is_rejected(missing_field):
    benchmark = make_valid_benchmark()
    del benchmark["queries"][0][missing_field]

    with pytest.raises(BenchmarkValidationError):
        validate_benchmark(benchmark)


@pytest.mark.parametrize(
    "invalid_grade",
    [-1, 3, "2", True],
)
def test_invalid_grade_is_rejected(invalid_grade):
    benchmark = make_valid_benchmark()
    benchmark["queries"][0]["judgments"] = {
        "paper-1": invalid_grade,
    }

    with pytest.raises(BenchmarkValidationError):
        validate_benchmark(benchmark)


def test_empty_query_text_is_rejected():
    benchmark = make_valid_benchmark()
    benchmark["queries"][0]["query"] = "   "

    with pytest.raises(BenchmarkValidationError):
        validate_benchmark(benchmark)


def test_invalid_split_is_rejected():
    benchmark = make_valid_benchmark()
    benchmark["queries"][0]["split"] = "training"

    with pytest.raises(BenchmarkValidationError):
        validate_benchmark(benchmark)


def test_both_splits_are_required():
    benchmark = make_valid_benchmark()

    for query in benchmark["queries"]:
        query["split"] = "dev"

    with pytest.raises(BenchmarkValidationError):
        validate_benchmark(benchmark)


def test_query_count_below_twenty_is_rejected():
    benchmark = make_valid_benchmark()
    benchmark["queries"].pop()

    with pytest.raises(BenchmarkValidationError):
        validate_benchmark(benchmark)


def test_unknown_paper_id_is_rejected():
    benchmark = make_valid_benchmark()
    valid_ids = {
        f"paper-{number}"
        for number in range(1, 21)
    }
    benchmark["queries"][0]["judgments"] = {
        "unknown-paper": 2,
    }

    with pytest.raises(BenchmarkValidationError):
        validate_benchmark(
            benchmark,
            valid_paper_ids=valid_ids,
        )


def test_queries_for_split():
    benchmark = make_valid_benchmark()

    assert len(queries_for_split(benchmark, "dev")) == 6
    assert len(queries_for_split(benchmark, "test")) == 14

    with pytest.raises(BenchmarkValidationError):
        queries_for_split(benchmark, "other")


def test_load_benchmark_from_file(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps(make_valid_benchmark()),
        encoding="utf-8",
    )

    loaded = load_benchmark(
        path,
        require_judgments=True,
    )

    assert loaded["benchmark_version"] == "1.0"
    assert len(loaded["queries"]) == 20


def test_duplicate_json_key_is_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"benchmark_version":"1.0",'
        '"benchmark_version":"2.0"}',
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkValidationError):
        load_benchmark(path)


def test_invalid_json_is_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(
        '{"benchmark_version":',
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkValidationError):
        load_benchmark(path)