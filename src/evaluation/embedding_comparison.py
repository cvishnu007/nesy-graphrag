"""Controlled retrieval comparison of locally cached embedding models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np

from src.evaluation.benchmark_io import load_benchmark, queries_for_split
from src.evaluation.ir_metrics import aggregate_query_metrics, evaluate_ranking


K_VALUES = (5, 10, 20)
MODEL_NAMES = {
    "specter": "sentence-transformers/allenai-specter",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


def _normalize(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Encoder output must be a two-dimensional matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Encoder returned a zero vector")
    return array / norms


def compare_embedding_models(
    benchmark: Mapping,
    corpus: list[dict],
    encoders: Mapping[str, object],
    *,
    split: str,
    top_k: int = 20,
) -> dict:
    if top_k < max(K_VALUES):
        raise ValueError(f"top_k must be at least {max(K_VALUES)}")
    queries = queries_for_split(benchmark, split)
    if not queries:
        raise ValueError(f"No queries for split {split}")
    corpus_index = {}
    for record in corpus:
        paper_id = record.get("id")
        if not isinstance(paper_id, str) or not paper_id:
            raise ValueError("Every corpus record requires a non-empty id")
        if paper_id in corpus_index:
            raise ValueError(f"Duplicate corpus ID: {paper_id}")
        corpus_index[paper_id] = record
    candidate_ids = sorted({paper_id for query in queries for paper_id in query["judgments"]})
    missing = sorted(set(candidate_ids) - set(corpus_index))
    if missing:
        raise ValueError(f"Judged paper IDs missing from corpus: {missing[:5]}")
    document_texts = [
        f"{corpus_index[paper_id].get('title', '')}. {corpus_index[paper_id].get('abstract', '')}".strip()
        for paper_id in candidate_ids
    ]
    query_ids = [query["query_id"] for query in queries]
    query_texts = [query["query"] for query in queries]
    models = {}
    all_rankings = []
    for model_name, encoder in encoders.items():
        started = perf_counter()
        document_vectors = _normalize(encoder.encode(
            document_texts, batch_size=32, show_progress_bar=False,
            convert_to_numpy=True,
        ))
        query_vectors = _normalize(encoder.encode(
            query_texts, batch_size=32, show_progress_bar=False,
            convert_to_numpy=True,
        ))
        metric_rows = []
        rankings = []
        for query, query_vector in zip(queries, query_vectors):
            scores = document_vectors @ query_vector
            order = np.argsort(-scores, kind="stable")
            ranked_ids = [candidate_ids[index] for index in order[:top_k]]
            metrics = evaluate_ranking(ranked_ids, query["judgments"], K_VALUES)
            metric_rows.append({"query_id": query["query_id"], **metrics})
            ranking = {
                "model": model_name,
                "query_id": query["query_id"],
                "split": split,
                "ranked_paper_ids": ranked_ids,
            }
            rankings.append(ranking)
            all_rankings.append(ranking)
        models[model_name] = {
            "query_ids": query_ids,
            "candidate_ids": candidate_ids,
            "runtime_seconds": perf_counter() - started,
            "summary": aggregate_query_metrics(metric_rows),
            "per_query_metrics": metric_rows,
        }
    return {
        "benchmark_version": benchmark["benchmark_version"],
        "split": split,
        "top_k": top_k,
        "models": models,
        "rankings": all_rankings,
        "reference_judgment_source": benchmark.get("judgment_metadata", {}).get("source", "unspecified"),
        "human_ground_truth": False,
        "controlled_variables": {
            "only_changed_component": "embedding_model",
            "identical_queries": True,
            "identical_candidate_universe": True,
            "identical_document_text": True,
            "similarity": "cosine",
            "k_values": list(K_VALUES),
            "production_chroma_modified": False,
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
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    import torch
    from sentence_transformers import SentenceTransformer

    benchmark = load_benchmark(args.benchmark, require_judgments=True)
    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoders = {
        alias: SentenceTransformer(name, local_files_only=True, device=device)
        for alias, name in MODEL_NAMES.items()
    }
    result = compare_embedding_models(
        benchmark, corpus, encoders, split=args.split, top_k=args.top_k
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rankings = result.pop("rankings")
    (output / "comparison_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "rankings.jsonl").open("w", encoding="utf-8") as file:
        for row in rankings:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "failures.jsonl").write_text("", encoding="utf-8")
    command = (
        ".\\venv\\Scripts\\python.exe -m src.evaluation.embedding_comparison "
        f"--benchmark {args.benchmark} --corpus {args.corpus} --split {args.split} "
        f"--top-k {args.top_k} --output-dir {args.output_dir}"
    )
    metadata = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "benchmark": args.benchmark,
        "benchmark_sha256": hashlib.sha256(Path(args.benchmark).read_bytes()).hexdigest(),
        "corpus": args.corpus,
        "corpus_sha256": hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
        "split": args.split,
        "models": MODEL_NAMES,
        "sentence_transformers_version": importlib.metadata.version("sentence-transformers"),
        "device": device,
        "failure_count": 0,
        "human_ground_truth": False,
        "controlled_variables": result["controlled_variables"],
        "reproduction_command": command,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.split == "dev":
        protocol = {
            "status": "frozen_after_development_run",
            "models": MODEL_NAMES,
            "similarity": "cosine",
            "top_k": args.top_k,
            "k_values": list(K_VALUES),
            "candidate_universe": "union of judged paper IDs for the split",
            "test_specific_selection": False,
        }
        (output / "comparison_protocol.json").write_text(
            json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({
        "split": args.split,
        "query_count": len(next(iter(result["models"].values()))["query_ids"]),
        "candidate_count": len(next(iter(result["models"].values()))["candidate_ids"]),
        "models": {
            name: data["summary"]["metrics"] for name, data in result["models"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
