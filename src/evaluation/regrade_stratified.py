"""Regrade the 44-query stratified pool with the 27B judge, a daily batch at a time.

The 7B judge behind RESULTS §2d misses the κ 0.6 bar on the ten non-GNN topics
(κ 0.57); the 27B judge reaches κ 0.88 there and on the GNN pairs (RESULTS §2b).
Regrading all 2,305 pooled pairs one paper per call takes about 12 days on Groq's
free tier (200K tokens/day), so this script grades a budget per run and resumes:

* Queries are graded in topic round-robin order (V2-01-1, V2-02-1, ..., V2-11-1,
  V2-01-2, ...), so a partial run already covers every topic.
* ``score`` uses only the queries whose pool is fully graded and reports, on the same
  queries, NDCG@10 under the 27B and the 7B labels and the predeclared comparisons.

Run on the machine with ``data/`` (the pool needs paper texts) and ``GROQ_API_KEY``:

    python -m src.evaluation.regrade_stratified grade --max-pairs 190   # once a day
    python -m src.evaluation.regrade_stratified score

Output: ``results/retrieval/stratified_v2_27b/`` (frozen 7B results are not touched).
The judge was validated at one paper per call on the non-GNN topics; keep
``--batch-size 1`` unless time forces 4 (validated on GNN pairs only, κ 0.88).
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path

FROZEN = Path("results/retrieval/stratified_v2_20261114")
OUT = Path("results/retrieval/stratified_v2_27b")
JUDGE = "groq:qwen/qwen3.8-27b@strict"


def round_robin(query_ids):
    """V2-TT-N ids ordered by N, then topic TT."""
    return sorted(query_ids, key=lambda q: (int(q.split("-")[2]), int(q.split("-")[1])))


def graded_labels(path):
    if not path.exists():
        return {}
    return {r["item_id"]: r["grade"] for r in map(json.loads, path.read_text().splitlines())}


def complete_queries(rows, graded):
    by_query = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row["item_id"] in graded)
    return sorted(q for q, done in by_query.items() if all(done))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("grade", "score"))
    parser.add_argument("--max-pairs", type=int, default=190, help="pairs to grade in this run")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--judge", default=JUDGE)
    args = parser.parse_args(argv)
    from src.evaluation.stratified_benchmark import COMPARISONS, QUERIES, load_rankings, pool

    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("rankings.jsonl", "queries_sha256.txt"):
        if not (OUT / name).exists():
            shutil.copy(FROZEN / name, OUT / name)
    queries = json.loads(Path(QUERIES).read_text(encoding="utf-8"))["queries"]
    rankings = load_rankings(OUT)
    labels_path = OUT / "labels.jsonl"

    if args.stage == "grade":
        from src.evaluation.judge_agreement import judge, make_client
        from src.utils.config import CLEAN_FILE

        corpus = {p["id"]: p for p in json.loads(Path(CLEAN_FILE).read_text(encoding="utf-8"))}
        rows = pool(queries, rankings, corpus)
        model_spec, prompt_name = args.judge.rsplit("@", 1)
        client = make_client(model_spec)
        budget = args.max_pairs
        by_query = defaultdict(list)
        for row in rows:
            by_query[row["query_id"]].append(row)
        log = json.loads((OUT / "judge.json").read_text()) if (OUT / "judge.json").exists() else {
            "judge": args.judge, "batch_size": args.batch_size, "temperature": 0.0, "pooled_pairs": len(rows),
            "order": "topic round-robin", "runs": []}
        before = len(graded_labels(labels_path))
        try:
            for query_id in round_robin(by_query):
                if budget <= 0:
                    break
                done = graded_labels(labels_path)
                todo = [r for r in by_query[query_id] if r["item_id"] not in done][:budget]
                if todo:
                    judge(client, todo, prompt_name, labels_path, batch_size=args.batch_size)
                    budget -= len(todo)
        except Exception as error:  # usually the daily token limit; progress is already saved
            print(f"stopped: {type(error).__name__}: {error}")
        graded = graded_labels(labels_path)
        log["runs"].append({"date": date.today().isoformat(), "graded": len(graded) - before, "total": len(graded)})
        (OUT / "judge.json").write_text(json.dumps(log, indent=2) + "\n")
        print(f"graded {len(graded)}/{len(rows)} pairs; {len(complete_queries(rows, graded))} of 44 queries complete")
        return 0

    from src.evaluation.regrade_pool import score
    from src.evaluation.significance import paired_permutation_test

    def as_labels(graded):
        out = defaultdict(dict)
        for item_id, grade in graded.items():
            query_id, paper_id = item_id.split("/", 1)
            out[query_id][paper_id] = grade
        return out

    new, old = graded_labels(labels_path), graded_labels(FROZEN / "labels.jsonl")
    rows = [{"query_id": i.split("/", 1)[0], "item_id": i} for i in old]  # the pool is the frozen pool
    ids = complete_queries(rows, new)
    if not ids:
        print("no query is fully graded yet")
        return 1
    report = {"judge": args.judge, "complete_queries": ids, "labels": {}}
    for name, graded in (("27b", new), ("7b", old)):
        summary = score(rankings, as_labels(graded), ids)
        tests = {}
        for a, b in COMPARISONS:
            deltas = [summary[a]["per_query_ndcg@10"][q] - summary[b]["per_query_ndcg@10"][q] for q in ids]
            result = paired_permutation_test(deltas) if len(deltas) > 1 else {"mean_delta": deltas[0]}
            tests[f"{a}_vs_{b}"] = {k: result[k] for k in ("mean_delta", "bootstrap_95_ci", "monte_carlo_two_sided_p")
                                    if k in result}
        report["labels"][name] = {"ndcg@10": {n: round(d["ndcg@10"], 4) for n, d in summary.items()},
                                  "comparisons": tests}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(ids)} complete queries")
    for system in sorted(report["labels"]["27b"]["ndcg@10"]):
        print(f"  {system:28} 27B {report['labels']['27b']['ndcg@10'][system]:.3f}   "
              f"7B {report['labels']['7b']['ndcg@10'][system]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
