"""Confirmatory retrieval evaluation on the topic-stratified 44-query set (v0.5).

The retrieval configuration was frozen on November 13, 2026 from the GNN
benchmark's dev split, and nothing is tuned on these queries, so all 44 are a
confirmation set (dev and test are also reported separately).

Stages, each resumable:

``rank``   Run variants and append rankings (top 20). The query encoder is global
           per process, so run SPECTER variants and BGE variants in separate
           processes with the matching environment (see ``FAMILIES``).
``grade``  Pool the union of every variant's top 20 per query and grade it with the
           human-validated judge (Qwen3.8-27B, strict prompt; kappa 0.88 held out).
           Papers are ordered by hash within each query, so batch position is
           unrelated to the method.
``score``  Fully judged metrics per variant, plus the predeclared paired tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

QUERIES = "evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json"
OUTPUT = "results/retrieval/stratified_v2_20261114"
DEPTH = 20
FAMILIES = {
    # Run with EMBEDDING_MODEL=allenai-specter CHROMA_COLLECTION=s2_papers EMBEDDING_QUERY_PREFIX=
    "specter": ["vector_abstract", "production_absolute", "bm25", "full_ta_bm25_similar"],
    # Run with the current .env (BGE)
    "bge": ["bge", "rrf_bge_bm25", "graph_relative_similar_bge", "full_bge_bm25_similar"],
}
EXPECTED_MODEL = {"specter": "allenai-specter", "bge": "BAAI/bge-base-en-v1.5"}
COMPARISONS = [
    ("full_bge_bm25_similar", "production_absolute"),
    ("bge", "production_absolute"),
    ("rrf_bge_bm25", "production_absolute"),
    ("bm25", "production_absolute"),
    ("full_bge_bm25_similar", "bm25"),
    ("full_bge_bm25_similar", "bge"),
    ("full_bge_bm25_similar", "rrf_bge_bm25"),
    ("graph_relative_similar_bge", "bge"),
    ("bge", "bm25"),
]


def variant_specs():
    from src.evaluation.variant_ablation import DEFAULT_VARIANTS

    specs = dict(DEFAULT_VARIANTS)
    specs.update(json.loads(Path("results/retrieval/variants_bge_20261113.json").read_text()))
    return specs


def rank(family, queries, output: Path):
    from src.evaluation.variant_ablation import build_variant
    from src.storage.neo4j_store import get_driver
    from src.utils import config

    if config.EMBEDDING_MODEL != EXPECTED_MODEL[family]:
        raise ValueError(f"family {family!r} needs EMBEDDING_MODEL={EXPECTED_MODEL[family]}, got {config.EMBEDDING_MODEL}")
    path = output / "rankings.jsonl"
    done = set()
    if path.exists():
        done = {(r["variant"], r["query_id"]) for r in map(json.loads, path.read_text().splitlines())}
    specs = variant_specs()
    driver = get_driver()
    try:
        with path.open("a") as stream:
            for name in FAMILIES[family]:
                retrieve = build_variant(specs[name], driver)
                for query in queries:
                    if (name, query["query_id"]) in done:
                        continue
                    ranked = []
                    for paper in retrieve(query["query"], DEPTH):
                        if paper["id"] not in ranked:
                            ranked.append(paper["id"])
                    stream.write(json.dumps({"variant": name, "family": family, "query_id": query["query_id"],
                                             "embedding_model": config.EMBEDDING_MODEL,
                                             "ranked_paper_ids": ranked[:DEPTH]}) + "\n")
                    stream.flush()
                print(f"ranked {name}", flush=True)
    finally:
        driver.close()


def load_rankings(output: Path):
    rankings = defaultdict(dict)
    for row in map(json.loads, (output / "rankings.jsonl").read_text().splitlines()):
        rankings[row["variant"]][row["query_id"]] = row["ranked_paper_ids"]
    return rankings


def pool(queries, rankings, corpus):
    rows = []
    for query in queries:
        ids = sorted({p for by_query in rankings.values() for p in by_query.get(query["query_id"], [])[:DEPTH]})
        for paper_id in ids:
            paper = corpus[paper_id]
            rows.append({"item_id": f"{query['query_id']}/{paper_id}", "query_id": query["query_id"],
                         "query": query["query"], "paper_id": paper_id,
                         "paper_title": paper.get("clean_title", paper.get("title", "")),
                         "paper_abstract": paper.get("clean_abstract", paper.get("abstract", ""))})
    return sorted(rows, key=lambda r: (r["query_id"], hashlib.sha256(r["item_id"].encode()).hexdigest()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("rank", "grade", "score"))
    parser.add_argument("--family", choices=sorted(FAMILIES))
    parser.add_argument("--queries", default=QUERIES)
    parser.add_argument("--output-dir", default=OUTPUT)
    parser.add_argument("--judge", default="groq:qwen/qwen3.8-27b@strict")
    parser.add_argument("--batch-size", type=int, default=4, help="papers per judge call (local 7B judge: 1)")
    args = parser.parse_args(argv)
    query_bytes = Path(args.queries).read_bytes()
    benchmark = json.loads(query_bytes)
    if benchmark["status"] != "reviewed_frozen_for_pooling":
        raise ValueError("Query set is not reviewed and frozen")
    queries = benchmark["queries"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "queries_sha256.txt"
    digest = hashlib.sha256(query_bytes.replace(b"\r\n", b"\n")).hexdigest()
    if manifest.exists() and manifest.read_text().strip() != digest:
        raise ValueError("Query file changed after pooling started")
    manifest.write_text(digest + "\n")

    if args.stage == "rank":
        if not args.family:
            parser.error("rank needs --family")
        rank(args.family, queries, output)
        return 0

    rankings = load_rankings(output)
    missing = [(n, q["query_id"]) for f in FAMILIES.values() for n in f for q in queries if q["query_id"] not in rankings.get(n, {})]
    if missing:
        raise ValueError(f"{len(missing)} variant/query rankings missing, e.g. {missing[:3]}")
    from src.utils.config import CLEAN_FILE

    corpus = {p["id"]: p for p in json.loads(Path(CLEAN_FILE).read_text(encoding="utf-8"))}
    rows = pool(queries, rankings, corpus)
    labels_path = output / "labels.jsonl"
    if args.stage == "grade":
        from src.evaluation.judge_agreement import judge, make_client

        model_spec, prompt_name = args.judge.rsplit("@", 1)
        judge(make_client(model_spec), rows, prompt_name, labels_path, batch_size=args.batch_size)
        (output / "judge.json").write_text(json.dumps({
            "judge": args.judge, "batch_size": args.batch_size, "pool_depth": DEPTH, "pooled_pairs": len(rows),
            "human_validation": ("results/retrieval/judge_agreement_groq_20261114/report.json" if args.judge.startswith("groq") else "results/retrieval/judge_agreement_local_20261114/report.json")}, indent=2) + "\n")
        print(f"graded {len(rows)} pooled pairs", flush=True)
        return 0

    from src.evaluation.regrade_pool import score
    from src.evaluation.significance import paired_significance, paired_permutation_test

    graded = {r["item_id"]: r["grade"] for r in map(json.loads, labels_path.read_text().splitlines())}
    labels = defaultdict(dict)
    for row in rows:
        labels[row["query_id"]][row["paper_id"]] = graded[row["item_id"]]
    report = {"queries": args.queries, "queries_sha256": digest, "judge": args.judge, "pooled_pairs": len(rows),
              "relevant_per_query": {q["query_id"]: {"grade2": sum(v == 2 for v in labels[q["query_id"]].values()),
                                                     "grade1": sum(v == 1 for v in labels[q["query_id"]].values())}
                                     for q in queries},
              "splits": {}}
    for split in ("all", "dev", "test"):
        ids = sorted(q["query_id"] for q in queries if split == "all" or q["split"] == split)
        summary = score(rankings, labels, ids)
        comparisons = {}
        for challenger, reference in COMPARISONS:
            a, b = summary[challenger]["per_query_ndcg@10"], summary[reference]["per_query_ndcg@10"]
            deltas = [a[q] - b[q] for q in ids]
            test = paired_significance(deltas) if len(deltas) <= 20 else paired_permutation_test(deltas)
            comparisons[f"{challenger}_vs_{reference}"] = {**test, "wins": sum(d > 0 for d in deltas),
                                                           "losses": sum(d < 0 for d in deltas)}
        report["splits"][split] = {"query_count": len(ids), "variants": summary, "comparisons": comparisons}
        print(f"\n{split} ({len(ids)} queries)")
        for name, data in sorted(summary.items(), key=lambda item: -item[1]["ndcg@10"]):
            print(f"  {name:28} ndcg@10={data['ndcg@10']:.3f} p@10={data['precision@10']:.3f} "
                  f"recall@20={data['recall@20']:.3f} unjudged@20={data['unjudged_rate@20']:.3f}")
        for name, data in comparisons.items():
            p = data.get("exact_two_sided_randomization_p", data.get("monte_carlo_two_sided_p"))
            print(f"  {name:50} delta={data['mean_delta']:+.3f} p={p:.4f} W/L={data['wins']}/{data['losses']}")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
