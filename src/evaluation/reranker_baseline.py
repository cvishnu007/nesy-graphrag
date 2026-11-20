"""Cross-encoder reranker baseline on the 44-query stratified benchmark (RESULTS §2d).

Hybrid search plus a cross-encoder reranker is today's standard alternative to graph
expansion, so the audit asked for it. Design (predeclared 2026-11-20, nothing tuned):

* Rerank the top 20 of two frozen runs: BGE + BM25 (``rrf_bge_bm25``) and the
  selected system with graph expansion (``full_bge_bm25_similar``).
* Depth 20 is the judged pool depth (the pool is the union of every system's top 20),
  so the reranked top 10 is fully judged and no new LLM grades are needed. A deeper
  rerank (50-100, the usual setting) would introduce unjudged papers; it is not run.
* Reranker: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (22M parameters, MS MARCO),
  chosen because it runs in well under 1 GB of memory on a laptop. ``--model`` accepts
  a larger cross-encoder such as ``BAAI/bge-reranker-base``.
* Input: the query, and the paper's title and abstract as returned by the Semantic
  Scholar API for the same paper IDs (original case). Texts are cached in ``data/``
  (not committed; refetch with ``fetch``), only rankings and scores are committed.
* Scoring with the frozen 7B judge labels and the project's metric code; paired
  sign-flip tests (20,000 permutations) with bootstrap 95% CIs.

    python -m src.evaluation.reranker_baseline fetch
    python -m src.evaluation.reranker_baseline rerank [--model cross-encoder/ms-marco-MiniLM-L-6-v2]
    python -m src.evaluation.reranker_baseline score
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE = Path("results/retrieval/stratified_v2_20261114")
QUERIES = Path("evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json")
OUT = Path("results/retrieval/reranker_20261120")
TEXTS = Path("data/reranker_texts_s2.jsonl")
SYSTEMS = ("rrf_bge_bm25", "full_bge_bm25_similar")
DEPTH = 20
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,abstract"


def variant_name(model, system):
    short = re.sub(r"[^a-z0-9]+", "_", model.split("/")[-1].lower()).strip("_")
    return f"rerank_{short}__{system}"


def load(base=BASE):
    from src.evaluation.stratified_benchmark import load_rankings

    queries = json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
    labels = defaultdict(dict)
    for row in map(json.loads, (base / "labels.jsonl").read_text().splitlines()):
        query_id, paper_id = row["item_id"].split("/", 1)
        labels[query_id][paper_id] = row["grade"]
    return queries, load_rankings(base), labels


def rerank_order(paper_ids, scores):
    """Paper IDs by descending cross-encoder score; ties keep the original order."""
    return [p for _, _, p in sorted(zip((-s for s in scores), range(len(paper_ids)), paper_ids))]


def fetch(ids, path=TEXTS, *, tries=8):
    have = {}
    if path.exists():
        have = {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    todo = sorted(set(ids) - set(have))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for start in range(0, len(todo), 500):
            chunk = todo[start:start + 500]
            body = json.dumps({"ids": chunk}).encode()
            for attempt in range(tries):
                request = urllib.request.Request(S2_BATCH, data=body, method="POST", headers={
                    "Content-Type": "application/json", "User-Agent": "reranker-baseline/1.0"})
                try:
                    papers = json.load(urllib.request.urlopen(request, timeout=60))
                    break
                except urllib.error.HTTPError as error:
                    if error.code != 429 or attempt == tries - 1:
                        raise
                    time.sleep(min(60, 5 * 2 ** attempt))
            for paper_id, paper in zip(chunk, papers):
                if paper:
                    row = {"id": paper_id, "title": paper.get("title") or "", "abstract": paper.get("abstract") or ""}
                    stream.write(json.dumps(row) + "\n")
                    have[paper_id] = row
            print(f"fetched {min(start + 500, len(todo))}/{len(todo)}", flush=True)
    return have


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("fetch", "rerank", "score"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu keeps memory low on a laptop; cuda or mps if free")
    args = parser.parse_args(argv)
    queries, rankings, labels = load()
    ids = {p for s in SYSTEMS for ranked in rankings[s].values() for p in ranked[:DEPTH]}

    if args.stage == "fetch":
        texts = fetch(ids)
        missing = ids - set(texts)
        empty = sum(1 for p in ids & set(texts) if not texts[p]["abstract"])
        print(f"{len(ids & set(texts))}/{len(ids)} papers with text; {len(missing)} missing; {empty} without abstract")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    rerank_path = OUT / "rankings.jsonl"
    if args.stage == "rerank":
        from sentence_transformers import CrossEncoder

        texts = {r["id"]: r for r in map(json.loads, TEXTS.read_text(encoding="utf-8").splitlines())}
        model = CrossEncoder(args.model, max_length=512, device=args.device)
        kept = [json.loads(line) for line in rerank_path.read_text().splitlines()] if rerank_path.exists() else []
        kept = [r for r in kept if r["model"] != args.model]
        for system in SYSTEMS:
            for query in queries:
                ranked = rankings[system][query["query_id"]][:DEPTH]
                docs = [f"{texts.get(p, {}).get('title', '')}. {texts.get(p, {}).get('abstract', '')}".strip(". ")
                        for p in ranked]
                scores = [float(s) for s in model.predict([(query["query"], d) for d in docs],
                                                          batch_size=args.batch_size, show_progress_bar=False)]
                kept.append({"variant": variant_name(args.model, system), "model": args.model, "base": system,
                             "query_id": query["query_id"], "ranked_paper_ids": rerank_order(ranked, scores),
                             "scores": [round(s, 4) for s in sorted(scores, reverse=True)],
                             "papers_without_text": sum(p not in texts for p in ranked)})
            print(f"reranked {system} with {args.model}", flush=True)
        rerank_path.write_text("".join(json.dumps(r) + "\n" for r in kept))
        return 0

    from src.evaluation.regrade_pool import score
    from src.evaluation.significance import paired_permutation_test

    reranked = defaultdict(dict)
    models = {}
    for row in map(json.loads, rerank_path.read_text().splitlines()):
        reranked[row["variant"]][row["query_id"]] = row["ranked_paper_ids"]
        models[row["variant"]] = (row["model"], row["base"])
    ids_sorted = sorted(q["query_id"] for q in queries)
    systems = {name: rankings[name] for name in (*SYSTEMS, "bm25", "bge")} | dict(reranked)
    summary = score(systems, labels, ids_sorted)
    comparisons = {}
    for variant, (model, base) in sorted(models.items()):
        for reference in (base, "full_bge_bm25_similar", "bm25"):
            if reference == variant:
                continue
            a, b = summary[variant]["per_query_ndcg@10"], summary[reference]["per_query_ndcg@10"]
            deltas = [a[q] - b[q] for q in ids_sorted]
            test = paired_permutation_test(deltas)
            comparisons[f"{variant}_vs_{reference}"] = {
                "mean_delta": round(test["mean_delta"], 4),
                "bootstrap_95_ci": [round(x, 4) for x in test["bootstrap_95_ci"]],
                "p": test["monte_carlo_two_sided_p"],
                "wins": sum(d > 0 for d in deltas), "losses": sum(d < 0 for d in deltas)}
    report = {"design": __doc__.split("\n\n")[1], "depth": DEPTH, "judge_labels": str(BASE / "labels.jsonl"),
              "texts": "Semantic Scholar API, fetched 2026-11-20 (not committed)",
              "variants": {n: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()
                               if not k.startswith("per_query")} for n, d in summary.items()},
              "comparisons": comparisons}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, data in sorted(report["variants"].items(), key=lambda item: -item[1]["ndcg@10"]):
        print(f"  {name:62} ndcg@10={data['ndcg@10']:.3f} p@10={data['precision@10']:.3f} "
              f"unjudged@20={data['unjudged_rate@20']:.3f}")
    for name, data in comparisons.items():
        print(f"  {name:80} delta={data['mean_delta']:+.4f} CI={data['bootstrap_95_ci']} p={data['p']:.4f} "
              f"W/L={data['wins']}/{data['losses']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
