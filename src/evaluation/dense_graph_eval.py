"""Does the densified citation graph help retrieval? Dev selection, then one test run.

Variants extend the selected system (BGE + BM25 + relative-gate graph expansion)
with the opt-in layers from ``src/ingestion/citation_densify.py``. Baselines come
from the stored stratified rankings (``results/retrieval/stratified_v2_20261114``).
Newly retrieved, unjudged papers are graded by the same human-validated judge
(local qwen2.5-7B, strict prompt, one paper per call) into a separate extension
file, so the frozen pool labels are never modified.

Predeclared rule (2026-11-15, before any dense result was scored): on the 22 dev
queries, choose the dense variant with the highest mean NDCG@10. Test it once on
the 22 test queries (``--final-test``) against ``full_bge_bm25_similar`` (current
system) and ``rrf_bge_bm25`` (no graph). A dense variant replaces the current
system only if it is better on test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

BASE = Path("results/retrieval/stratified_v2_20261114")
OUT = Path("results/retrieval/dense_graph_20261115")
QUERIES = "evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json"
GRAPH = {"neural": True, "graph": "relative", "margin": 0.03, "limit": 50, "lexical_weight": 1.0}
VARIANTS = {
    "dense_all": {**GRAPH, "edges": ["CITES", "CITES_ADDED", "SIMILAR_TO", "COUPLED_WITH"]},
    "dense_citation_only": {**GRAPH, "edges": ["CITES", "CITES_ADDED", "COUPLED_WITH"]},
    "dense_cites_similar": {**GRAPH, "edges": ["CITES", "CITES_ADDED", "SIMILAR_TO"]},
}
BASELINES = ("full_bge_bm25_similar", "rrf_bge_bm25", "bge", "bm25", "production_absolute")


def labels():
    graded = {}
    for path in (BASE / "labels.jsonl", OUT / "labels_extension.jsonl"):
        if path.exists():
            for row in map(json.loads, path.read_text().splitlines()):
                graded[row["item_id"]] = row["grade"]
    return graded


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.final_test:
        parser.error("The test split needs --final-test, after the dev selection is frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    selection_path = OUT / "selection.json"
    variants = dict(VARIANTS)
    if args.split == "test":
        if not selection_path.exists():
            parser.error("Run dev first; selection.json is missing")
        chosen = json.loads(selection_path.read_text())["selected"]
        variants = {chosen: VARIANTS[chosen]}
    from src.evaluation.judge_agreement import judge, make_client
    from src.evaluation.regrade_pool import score
    from src.evaluation.significance import paired_permutation_test, paired_significance
    from src.evaluation.variant_ablation import build_variant
    from src.storage.neo4j_store import get_driver
    from src.utils.config import CLEAN_FILE

    queries = [q for q in json.loads(Path(QUERIES).read_text(encoding="utf-8"))["queries"] if q["split"] == args.split]
    rankings_path = OUT / f"rankings_{args.split}.jsonl"
    rankings = defaultdict(dict)
    if rankings_path.exists():
        for row in map(json.loads, rankings_path.read_text().splitlines()):
            rankings[row["variant"]][row["query_id"]] = row["ranked_paper_ids"]
    driver = get_driver()
    try:
        with rankings_path.open("a") as stream:
            for name, spec in variants.items():
                retrieve = build_variant(spec, driver)
                for query in queries:
                    if query["query_id"] in rankings[name]:
                        continue
                    ranked = []
                    for paper in retrieve(query["query"], 20):
                        if paper["id"] not in ranked:
                            ranked.append(paper["id"])
                    rankings[name][query["query_id"]] = ranked[:20]
                    stream.write(json.dumps({"variant": name, "query_id": query["query_id"], "spec": spec,
                                             "ranked_paper_ids": ranked[:20]}) + "\n")
                print(f"ranked {name}", flush=True)
    finally:
        driver.close()

    graded = labels()
    corpus = {p["id"]: p for p in json.loads(Path(CLEAN_FILE).read_text(encoding="utf-8"))}
    text = {q["query_id"]: q["query"] for q in queries}
    todo = sorted({(q, p) for name in variants for q, ids in rankings[name].items() for p in ids
                   if f"{q}/{p}" not in graded},
                  key=lambda x: (x[0], hashlib.sha256(f"{x[0]}/{x[1]}".encode()).hexdigest()))
    print(f"{len(todo)} new pairs to grade", flush=True)
    if todo:
        rows = [{"item_id": f"{q}/{p}", "query": text[q], "paper_title": corpus[p]["clean_title"],
                 "paper_abstract": corpus[p]["clean_abstract"]} for q, p in todo]
        judge(make_client("ollama:qwen2.5:7b-instruct"), rows, "strict", OUT / "labels_extension.jsonl", batch_size=1)
        graded = labels()

    for row in map(json.loads, (BASE / "rankings.jsonl").read_text().splitlines()):
        if row["variant"] in BASELINES and row["query_id"] in text:
            rankings[row["variant"]][row["query_id"]] = row["ranked_paper_ids"]
    by_query = defaultdict(dict)
    for item_id, grade in graded.items():
        query_id, paper_id = item_id.split("/", 1)
        by_query[query_id][paper_id] = grade
    ids = sorted(text)
    summary = score({n: rankings[n] for n in [*variants, *BASELINES]}, by_query, ids)
    comparisons = {}
    for challenger in variants:
        for reference in ("full_bge_bm25_similar", "rrf_bge_bm25", "bge"):
            a, b = summary[challenger]["per_query_ndcg@10"], summary[reference]["per_query_ndcg@10"]
            deltas = [a[q] - b[q] for q in ids]
            test = paired_significance(deltas) if len(deltas) <= 20 else paired_permutation_test(deltas)
            comparisons[f"{challenger}_vs_{reference}"] = {**test, "p": test.get("exact_two_sided_randomization_p", test.get("monte_carlo_two_sided_p")),
                                                          "wins": sum(d > 0 for d in deltas),
                                                          "losses": sum(d < 0 for d in deltas)}
    report = {"split": args.split, "rule": __doc__.split("Predeclared rule")[1].strip(), "variants": summary,
              "comparisons": comparisons, "extension_pairs_graded": len(todo)}
    if args.split == "dev":
        report["selected"] = max(sorted(variants), key=lambda n: summary[n]["ndcg@10"])
        selection_path.write_text(json.dumps({"selected": report["selected"], "spec": VARIANTS[report["selected"]],
                                              "dev_ndcg@10": {n: summary[n]["ndcg@10"] for n in variants}}, indent=2) + "\n")
    (OUT / f"summary_{args.split}.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, data in sorted(summary.items(), key=lambda item: -item[1]["ndcg@10"]):
        print(f"  {name:24} ndcg@10={data['ndcg@10']:.3f} p@10={data['precision@10']:.3f} "
              f"recall@20={data['recall@20']:.3f} unjudged@20={data['unjudged_rate@20']:.3f}")
    for name, data in comparisons.items():
        print(f"  {name:48} delta={data['mean_delta']:+.3f} p={data['p']:.4f} "
              f"W/L={data['wins']}/{data['losses']}")
    if "selected" in report:
        print("selected on dev:", report["selected"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
