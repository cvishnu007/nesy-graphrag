"""Extend a frozen judgment pool with documents retrieved by new variants.

New retrieval variants surface documents the original pool never judged. This
module grades exactly those documents with the same blinded judge, prompt, and
settings as the frozen pool (``refresh_retrieval.grade_candidates``), writing a
separate checkpoint so the frozen labels are never modified. It then rescores
the variants with frozen + extension judgments, so standard metrics no longer
treat unjudged documents as irrelevant.

Grading is restricted to one split. Run it on dev for configuration selection;
extend test only once, after the configuration is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from src.evaluation.ir_metrics import aggregate_query_metrics, evaluate_ranking

K_VALUES = (5, 10, 20)


def unjudged_rows(benchmark, rankings, corpus, *, split, depth=20):
    """Return method-hidden candidate rows for unjudged retrieved documents."""
    queries = {q["query_id"]: q for q in benchmark["queries"] if q["split"] == split}
    needed = defaultdict(set)
    for row in rankings:
        query = queries.get(row["query_id"])
        if query is None:
            continue
        for paper_id in row["ranked_paper_ids"][:depth]:
            if paper_id not in query["judgments"]:
                needed[row["query_id"]].add(paper_id)
    rows = []
    for query_id in sorted(needed):
        for paper_id in sorted(needed[query_id]):
            paper = corpus.get(paper_id)
            if paper is None:
                raise ValueError(f"Retrieved paper missing from corpus: {paper_id}")
            rows.append({
                "query_id": query_id, "split": split, "query": queries[query_id]["query"],
                "paper_id": paper_id,
                "title": paper.get("clean_title", paper.get("title", "")),
                "abstract": paper.get("clean_abstract", paper.get("abstract", "")),
            })
    return rows


def merged_judgments(benchmark, extension):
    merged = {}
    for query in benchmark["queries"]:
        labels = dict(query["judgments"])
        for key, value in extension.items():
            query_id, paper_id = key.split("/", 1)
            if query_id == query["query_id"]:
                if paper_id in labels:
                    raise ValueError(f"Extension relabels a frozen judgment: {key}")
                labels[paper_id] = value["grade"]
        merged[query["query_id"]] = labels
    return merged


def rescore(benchmark, rankings, judgments, *, split):
    rows = defaultdict(list)
    for row in rankings:
        query_id = row["query_id"]
        if query_id not in judgments or not any(q["query_id"] == query_id and q["split"] == split
                                                for q in benchmark["queries"]):
            continue
        rows[row["variant"]].append({"query_id": query_id,
                                     **evaluate_ranking(row["ranked_paper_ids"], judgments[query_id], K_VALUES)})
    summary = {}
    for variant, values in rows.items():
        metrics = aggregate_query_metrics(values)["metrics"]
        summary[variant] = {
            "query_count": len(values),
            **{key: metrics[key]["mean"] for key in ("ndcg@10", "precision@10", "recall@20", "map", "mrr")},
            "unjudged_rate@20": metrics["unjudged_rate@20"]["mean"],
            "per_query_ndcg@10": {value["query_id"]: value["ndcg@10"] for value in values},
        }
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ablation-dir", required=True, help="variant_ablation output directory")
    parser.add_argument("--benchmark", default="results/retrieval/expanded_20260913/benchmark.json")
    parser.add_argument("--judge-protocol", default="results/retrieval/expanded_20260913/judge_protocol.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    if args.split == "test" and not args.final_test:
        parser.error("Extending the test pool requires --final-test")
    from src.evaluation.refresh_retrieval import JUDGE_VERSION, grade_candidates
    from src.utils.config import CLEAN_FILE
    from src.utils.llm_client import create_llm_client, llm_runtime_metadata

    directory = Path(args.ablation_dir)
    benchmark = json.loads(Path(args.benchmark).read_text())
    rankings = [json.loads(line) for line in (directory / "rankings.jsonl").read_text().splitlines() if line.strip()]
    corpus = {paper["id"]: paper for paper in json.loads(Path(CLEAN_FILE).read_text(encoding="utf-8"))}
    rows = unjudged_rows(benchmark, rankings, corpus, split=args.split)

    frozen = json.loads(Path(args.judge_protocol).read_text())
    metadata = {**llm_runtime_metadata(), "prompt_version": JUDGE_VERSION, "batch_size": args.batch_size,
                "temperature": 0.0, "max_tokens": max(64, args.batch_size * 4 + 16),
                "structured_output": frozen.get("structured_output")}
    if metadata != frozen:
        raise ValueError(f"Judge settings differ from the frozen pool: {metadata} vs {frozen}")
    checkpoint = directory / f"extension_{args.split}_checkpoint.json"
    grades = grade_candidates(rows, create_llm_client(), checkpoint, batch_size=args.batch_size, structured=True)
    extension = {key: value for key, value in grades.items()}
    summary = rescore(benchmark, rankings, merged_judgments(benchmark, extension), split=args.split)
    report = {
        "split": args.split, "extension_judgments": len(extension),
        "judge": metadata, "label_source": "local Ollama AI-reference labels (same judge as frozen pool)",
        "frozen_benchmark_sha256": hashlib.sha256(Path(args.benchmark).read_bytes()).hexdigest(),
        "variants": summary,
    }
    (directory / f"extended_{args.split}_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, data in sorted(summary.items(), key=lambda item: -item[1]["ndcg@10"]):
        print(f"{name:26} ndcg@10={data['ndcg@10']:.4f} p@10={data['precision@10']:.4f} "
              f"recall@20={data['recall@20']:.4f} map={data['map']:.4f} unjudged@20={data['unjudged_rate@20']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
