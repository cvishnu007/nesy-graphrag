"""Regrade the whole project relevance pool with the human-validated judge.

The frozen pool labels (local qwen2.5-3B) agree poorly with human grades
(quadratic-weighted kappa 0.17 on 200 human-graded pairs) and show a strong
batch-position bias. ``judge_agreement.py`` selected Qwen3.8-27B with the
``strict`` prompt on one half of the human grades and confirmed it on the other
half (kappa 0.88 on both). This module grades every pooled pair, i.e. the frozen
pool plus all pool extensions, with that judge, into a separate label file, and
rescores every stored ranking. Frozen labels and rankings are never modified.

Within each query, papers are ordered by a hash before batching, so a paper's
batch position is unrelated to the method that retrieved it.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from src.evaluation.ir_metrics import aggregate_query_metrics, evaluate_ranking
from src.evaluation.judge_agreement import judge, make_client
from src.evaluation.significance import paired_significance

K_VALUES = (5, 10, 20)
BENCHMARK = "results/retrieval/expanded_20260913/benchmark.json"
CANDIDATES = "results/retrieval/expanded_20260913/candidates.csv"
EXTENSIONS = "results/retrieval/variant_ablation*/extension_*_checkpoint.json"
RANKING_FILES = {
    "original_methods": "results/retrieval/expanded_20260913/{split}/rankings.jsonl",
    "specter_variants": "results/retrieval/variant_ablation_{split}_20261113/rankings.jsonl",
    "bge_variants": "results/retrieval/variant_ablation_bge_{split}_20261113/rankings.jsonl",
}
COMPARISONS = [  # (challenger, reference), predeclared on Nov 13 (selection.json)
    ("full_bge_bm25_similar", "production_absolute"),
    ("bge", "production_absolute"),
    ("rrf_bge_bm25", "production_absolute"),
    ("bm25", "production_absolute"),
    ("full_bge_bm25_similar", "bm25"),
    ("full_bge_bm25_similar", "bge"),
    ("bge", "bm25"),
]


def pool_rows(benchmark, candidates_path=CANDIDATES, extension_glob=EXTENSIONS, clean_file=None):
    """Every judged (query, paper) pair with its text, ordered by hash within each query."""
    queries = {q["query_id"]: q for q in benchmark["queries"]}
    text = {}
    with open(candidates_path, encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            text[row["paper_id"]] = (row["title"], row["abstract"])
    pairs = {(q["query_id"], p) for q in benchmark["queries"] for p in q["judgments"]}
    for path in sorted(glob.glob(extension_glob)):
        data = json.loads(Path(path).read_text())
        pairs |= {tuple(key.split("/", 1)) for key in data.get("judgments", data)}
    missing = {p for _, p in pairs if p not in text}
    if missing:
        from src.utils.config import CLEAN_FILE

        for paper in json.loads(Path(clean_file or CLEAN_FILE).read_text(encoding="utf-8")):
            if paper["id"] in missing:
                text[paper["id"]] = (paper.get("clean_title", paper.get("title", "")),
                                     paper.get("clean_abstract", paper.get("abstract", "")))
    absent = sorted({p for _, p in pairs if p not in text})
    if absent:
        raise ValueError(f"{len(absent)} pooled papers have no text, e.g. {absent[:3]}")
    rows = [{"item_id": f"{q}/{p}", "query_id": q, "split": queries[q]["split"], "query": queries[q]["query"],
             "paper_id": p, "paper_title": text[p][0], "paper_abstract": text[p][1]} for q, p in pairs]
    return sorted(rows, key=lambda r: (r["query_id"], hashlib.sha256(r["item_id"].encode()).hexdigest()))


def load_rankings(split):
    rankings = defaultdict(dict)
    for family, pattern in RANKING_FILES.items():
        path = Path(pattern.format(split=split))
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                name = row.get("variant") or row["method"]
                if family == "original_methods":
                    name = f"orig_{name}"
                rankings[name][row["query_id"]] = row["ranked_paper_ids"]
    return rankings


def score(rankings, labels, query_ids):
    summary = {}
    for name, by_query in sorted(rankings.items()):
        values = [{"query_id": q, **evaluate_ranking(by_query[q], labels[q], K_VALUES)}
                  for q in query_ids if q in by_query]
        metrics = aggregate_query_metrics(values)["metrics"]
        summary[name] = {"query_count": len(values),
                         **{k: metrics[k]["mean"] for k in ("ndcg@10", "precision@10", "recall@20", "map", "mrr")},
                         "unjudged_rate@20": metrics["unjudged_rate@20"]["mean"],
                         "per_query_ndcg@10": {v["query_id"]: v["ndcg@10"] for v in values}}
    return summary


def compare(summary):
    results = {}
    for challenger, reference in COMPARISONS:
        if challenger not in summary or reference not in summary:
            continue
        a, b = summary[challenger]["per_query_ndcg@10"], summary[reference]["per_query_ndcg@10"]
        deltas = [a[q] - b[q] for q in sorted(a) if q in b]
        test = paired_significance(deltas)
        results[f"{challenger}_vs_{reference}"] = {
            **test, "wins": sum(d > 0 for d in deltas), "ties": sum(d == 0 for d in deltas)}
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", default="groq:qwen/qwen3.8-27b@strict")
    parser.add_argument("--batch-size", type=int, default=4, help="papers per judge call (local 7B judge: 1)")
    parser.add_argument("--output-dir", default="results/retrieval/regraded_qwen3_8_27b_strict_20261114")
    parser.add_argument("--grade", action="store_true", help="call the judge for ungraded pairs (otherwise score only)")
    args = parser.parse_args(argv)
    benchmark = json.loads(Path(BENCHMARK).read_text())
    rows = pool_rows(benchmark)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "labels.jsonl"
    model_spec, prompt_name = args.judge.rsplit("@", 1)
    if args.grade:
        judge(make_client(model_spec), rows, prompt_name, checkpoint, batch_size=args.batch_size)
    graded = {r["item_id"]: r["grade"] for r in map(json.loads, checkpoint.read_text().splitlines())} \
        if checkpoint.exists() else {}
    print(f"graded {sum(r['item_id'] in graded for r in rows)} of {len(rows)} pooled pairs", flush=True)
    if any(r["item_id"] not in graded for r in rows):
        return 1
    labels = defaultdict(dict)
    for row in rows:
        labels[row["query_id"]][row["paper_id"]] = graded[row["item_id"]]
    frozen_labels = {q["query_id"]: q["judgments"] for q in benchmark["queries"]}
    old = [frozen_labels[r["query_id"]][r["paper_id"]] for r in rows if r["paper_id"] in frozen_labels[r["query_id"]]]
    new = [graded[r["item_id"]] for r in rows if r["paper_id"] in frozen_labels[r["query_id"]]]
    from src.evaluation.human_agreement import pair_report

    report = {"judge": args.judge, "pooled_pairs": len(rows),
              "grade_counts": {g: sum(v == g for v in graded.values()) for g in (0, 1, 2)},
              "agreement_with_frozen_3b_labels": pair_report(old, new),
              "human_validation": ("results/retrieval/judge_agreement_groq_20261114/report.json" if args.judge.startswith("groq") else "results/retrieval/judge_agreement_local_20261114/report.json"),
              "splits": {}}
    for split in ("dev", "test"):
        query_ids = sorted(q["query_id"] for q in benchmark["queries"] if q["split"] == split)
        summary = score(load_rankings(split), labels, query_ids)
        report["splits"][split] = {"variants": summary, "comparisons": compare(summary)}
        print(f"\n{split} ({len(query_ids)} queries)")
        for name, data in sorted(summary.items(), key=lambda item: -item[1]["ndcg@10"]):
            print(f"  {name:30} ndcg@10={data['ndcg@10']:.3f} p@10={data['precision@10']:.3f} "
                  f"map={data['map']:.3f} unjudged@20={data['unjudged_rate@20']:.3f}")
        for name, data in report["splits"][split]["comparisons"].items():
            print(f"  {name:50} delta={data['mean_delta']:+.3f} p={data['exact_two_sided_randomization_p']:.4f} "
                  f"wins={data['wins']}/{data['pair_count']}")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
