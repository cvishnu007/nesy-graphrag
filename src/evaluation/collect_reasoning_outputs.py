"""Collect real frozen-query pipeline outputs for Phase 2 annotation pools.

Contradiction collection can use graph candidates alone or the existing blinded
LLM prediction path. Support and hypothesis collection use the existing production
review/hypothesis functions and therefore require configured Groq access.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from src.evaluation.benchmark_io import load_benchmark, queries_for_split


TASKS = ("contradiction", "support", "hypothesis")


def _has_llm_failure(task: str, result: Mapping) -> bool:
    """Detect pipeline functions that caught an LLM error and returned it as data."""
    if task == "support":
        answers = result.get("raw_answers") or [result.get("raw_answer")]
        return any(str(answer or "").startswith("LLM call failed:") for answer in answers)
    if task == "hypothesis":
        items = [
            *(result.get("hypotheses", []) or []),
            *(result.get("rejected_hypotheses", []) or []),
        ]
        return any(
            str(item.get("llm_hypothesis") or "").startswith("LLM call failed:")
            for item in items if isinstance(item, Mapping)
        )
    if task == "contradiction":
        return any(
            str(item.get("llm_analysis") or "").startswith("LLM call failed:")
            for item in (result.get("contradictions", []) or []) if isinstance(item, Mapping)
        )
    return False


def collect_with_functions(
    queries: Iterable[Mapping],
    tasks: list[str],
    functions: Mapping[str, Callable[[str, int], dict | list]],
    *,
    top_k: int,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Collect in deterministic order; injected functions keep tests offline."""
    if not tasks or any(task not in TASKS for task in tasks):
        raise ValueError(f"tasks must contain values from {TASKS}")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    outputs = {task: [] for task in tasks}
    failures = []
    ordered_queries = sorted(queries, key=lambda item: item["query_id"])
    for query in ordered_queries:
        for task in tasks:
            try:
                result = functions[task](query["query"], top_k)
                if task == "contradiction" and isinstance(result, list):
                    result = {"query": query["query"], "contradictions": result}
                if not isinstance(result, dict):
                    raise TypeError("pipeline task did not return an object")
                if _has_llm_failure(task, result):
                    raise RuntimeError("pipeline returned a caught LLM failure")
                outputs[task].append({
                    **result,
                    "query_id": query["query_id"],
                    "split": query["split"],
                    "query": query["query"],
                })
            except Exception as error:
                failures.append({
                    "task": task, "query_id": query.get("query_id"),
                    "error_type": type(error).__name__,
                    "message": "Pipeline collection failed; inspect local logs without storing secrets",
                })
    return outputs, failures


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--benchmark", default="evaluation/benchmarks/retrieval_queries_judged.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--contradiction-mode", choices=("candidates", "llm"), default="candidates",
        help="Use graph candidates only or retain LLM verdicts in the protected sidecar",
    )
    parser.add_argument("--llm-model-override")
    parser.add_argument("--llm-fallback-override")
    parser.add_argument(
        "--configuration-label",
        help="Required audit label whenever an LLM model override is supplied",
    )
    arguments = parser.parse_args(argv)
    if len(set(arguments.tasks)) != len(arguments.tasks):
        parser.error("--tasks values must be unique")
    has_override = bool(arguments.llm_model_override or arguments.llm_fallback_override)
    if has_override and not (arguments.configuration_label or "").strip():
        parser.error("--configuration-label is required with an LLM model override")
    if arguments.configuration_label and not has_override:
        parser.error("--configuration-label requires an LLM model override")
    if arguments.llm_model_override:
        os.environ["LLM_MODEL"] = arguments.llm_model_override
    if arguments.llm_fallback_override:
        os.environ["LLM_MODEL_FALLBACK"] = arguments.llm_fallback_override
    directory = Path(arguments.output_dir)
    if directory.exists() and any(directory.iterdir()) and not arguments.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    benchmark = load_benchmark(arguments.benchmark, require_judgments=False)
    queries = queries_for_split(benchmark, arguments.split)

    from src.pipeline.contradiction import detect_contradictions, llm_contradict
    from src.pipeline.hypothesis import llm_hypothesis
    from src.pipeline.review import llm_review
    from src.storage.neo4j_store import get_driver

    driver = get_driver()
    groq_client = None
    try:
        needs_groq = any(task in {"support", "hypothesis"} for task in arguments.tasks) or (
            "contradiction" in arguments.tasks and arguments.contradiction_mode == "llm"
        )
        if needs_groq:
            from groq import Groq
            from src.utils.config import (
                GROQ_API_KEY, LLM_MODEL, LLM_MODEL_FALLBACK, is_configured,
            )

            if not is_configured(GROQ_API_KEY):
                raise RuntimeError("Groq is required for support/hypothesis collection but is not configured")
            groq_client = Groq(api_key=GROQ_API_KEY)
        functions = {
            "contradiction": lambda query, top_k: (
                llm_contradict(groq_client, driver, query, top_k=top_k)
                if arguments.contradiction_mode == "llm"
                else detect_contradictions(driver, query, top_k=top_k)
            ),
            "support": lambda query, top_k: llm_review(groq_client, driver, query, top_k=top_k),
            "hypothesis": lambda query, top_k: llm_hypothesis(groq_client, driver, query, top_k=top_k),
        }
        outputs, failures = collect_with_functions(
            queries, arguments.tasks, functions, top_k=arguments.top_k
        )
    finally:
        driver.close()

    if needs_groq:
        model_configuration = {
            "primary": LLM_MODEL,
            "fallback": LLM_MODEL_FALLBACK,
            "configuration_label": arguments.configuration_label or "phase1-frozen",
            "revised_configuration": has_override,
        }
    else:
        model_configuration = None
    for task, records in outputs.items():
        if model_configuration is not None:
            for record in records:
                record["generation_configuration"] = dict(model_configuration)
        _write_jsonl(directory / f"{task}_pipeline_outputs.jsonl", records)
    _write_jsonl(directory / "failures.jsonl", failures)
    summary = {
        "split": arguments.split,
        "query_count": len(queries),
        "task_output_counts": {task: len(records) for task, records in outputs.items()},
        "failure_count": len(failures),
        "generation_configuration": model_configuration,
    }
    (directory / "collection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
