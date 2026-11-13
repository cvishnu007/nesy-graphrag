"""Development-split ablation of retrieval variants against frozen judgments.

Every variant is built from public pipeline functions with explicit parameters,
so variants never depend on process-wide environment settings. Scores are
reported two ways because new variants may retrieve documents outside the
judged pool:

* standard: unjudged documents count as non-relevant (a lower bound);
* condensed: unjudged documents are removed before scoring (Sakai, 2007).

The unjudged rate is always reported. The test split is refused unless
``--final-test`` is given, so configuration choices are made on dev only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from src.evaluation.ir_metrics import (
    aggregate_query_metrics,
    condensed_ranking,
    evaluate_ranking,
)

K_VALUES = (5, 10, 20)

# name -> spec. collection None means the configured historical collection.
DEFAULT_VARIANTS = {
    "vector_abstract": {"neural": True},
    "vector_title_abstract": {"neural": True, "collection": "s2_papers_specter_ta"},
    "bm25": {"neural": False, "lexical_weight": 1.0},
    "production_absolute": {"neural": True, "graph": "absolute"},
    "rrf_vector_bm25": {"neural": True, "lexical_weight": 1.0},
    "rrf_ta_bm25": {"neural": True, "collection": "s2_papers_specter_ta", "lexical_weight": 1.0},
    "graph_relative_cites": {"neural": True, "graph": "relative", "margin": 0.03, "edges": ["CITES"]},
    "graph_relative_similar": {"neural": True, "graph": "relative", "margin": 0.03,
                               "edges": ["CITES", "SIMILAR_TO"], "limit": 50},
    "full_ta_bm25_similar": {"neural": True, "collection": "s2_papers_specter_ta", "graph": "relative",
                             "margin": 0.03, "edges": ["CITES", "SIMILAR_TO"], "limit": 50,
                             "lexical_weight": 1.0},
}


def build_variant(spec: Mapping, driver) -> Callable[[str, int], list[dict]]:
    """Return a retrieval function for one variant specification."""
    from src.pipeline.retrieval import (
        filter_symbolic_candidates, fuse_results, lexical_retrieve, symbolic_expand,
    )
    from src.storage.chroma_store import query as chroma_query, score_papers_against_query

    collection = spec.get("collection")
    lexical_weight = float(spec.get("lexical_weight", 0.0))
    graph_mode = spec.get("graph")

    def retrieve(query: str, top_k: int) -> list[dict]:
        neural = chroma_query(query, top_k=top_k, collection_name=collection) if spec.get("neural", True) else []
        for paper in neural:
            paper["source"] = "neural"
        symbolic = []
        if graph_mode:
            raw = symbolic_expand(driver, [p["id"] for p in neural],
                                  edge_types=spec.get("edges", ["CITES"]), limit=spec.get("limit", 20))
            scores = score_papers_against_query(query, [p["id"] for p in raw], collection_name=collection)
            symbolic = filter_symbolic_candidates(
                query, raw, scores, mode=graph_mode,
                seed_scores=[p["score"] for p in neural], margin=spec.get("margin"),
            )
        lexical = lexical_retrieve(query, top_k) if lexical_weight > 0 else None
        if not neural and not symbolic:
            return list(lexical or [])[:top_k]
        return fuse_results(neural, symbolic, top_k, lexical_papers=lexical, lexical_weight=lexical_weight)

    return retrieve


def score_variants(benchmark: Mapping, retrievers: Mapping[str, Callable], *, split: str,
                   top_k: int = 20) -> tuple[dict, list[dict]]:
    if top_k < max(K_VALUES):
        raise ValueError(f"top_k must be at least {max(K_VALUES)}")
    queries = [q for q in benchmark["queries"] if q["split"] == split]
    if not queries:
        raise ValueError(f"No queries for split {split}")
    rows = defaultdict(lambda: {"standard": [], "condensed": []})
    rankings = []
    for query in queries:
        judgments = query["judgments"]
        for name, retrieve in retrievers.items():
            ranked = []
            for paper in retrieve(query["query"], top_k):
                if paper["id"] not in ranked:
                    ranked.append(paper["id"])
            standard = evaluate_ranking(ranked, judgments, K_VALUES)
            condensed = evaluate_ranking(condensed_ranking(ranked, judgments), judgments, K_VALUES)
            rows[name]["standard"].append({"query_id": query["query_id"], **standard})
            rows[name]["condensed"].append({"query_id": query["query_id"], **condensed})
            rankings.append({"variant": name, "query_id": query["query_id"], "ranked_paper_ids": ranked})
    summary = {}
    for name, data in rows.items():
        standard = aggregate_query_metrics(data["standard"])["metrics"]
        condensed = aggregate_query_metrics(data["condensed"])["metrics"]
        summary[name] = {
            "query_count": len(data["standard"]),
            "standard": {key: standard[key]["mean"] for key in ("ndcg@10", "precision@10", "recall@20", "map", "mrr")},
            "condensed": {key: condensed[key]["mean"] for key in ("ndcg@10", "precision@10", "recall@20", "map", "mrr")},
            "unjudged_rate@10": standard["unjudged_rate@10"]["mean"],
            "unjudged_rate@20": standard["unjudged_rate@20"]["mean"],
            "per_query_condensed_ndcg@10": {row["query_id"]: row["ndcg@10"] for row in data["condensed"]},
        }
    return summary, rankings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", default="results/retrieval/expanded_20260913/benchmark.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--final-test", action="store_true",
                        help="Required for the test split; run once, after dev selection is frozen")
    parser.add_argument("--variants", nargs="+", help="Subset of variant names")
    parser.add_argument("--variant-file", help="JSON mapping of variant name to spec (overrides defaults)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
    if args.split == "test" and not args.final_test:
        parser.error("Test-split scoring requires --final-test; select configurations on dev")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    specs = json.loads(Path(args.variant_file).read_text()) if args.variant_file else dict(DEFAULT_VARIANTS)
    if args.variants:
        unknown = set(args.variants) - set(specs)
        if unknown:
            parser.error(f"Unknown variants: {sorted(unknown)}")
        specs = {name: specs[name] for name in args.variants}
    from src.storage.neo4j_store import get_driver
    from src.utils.config import CLEAN_FILE

    benchmark_bytes = Path(args.benchmark).read_bytes()
    benchmark = json.loads(benchmark_bytes)
    driver = get_driver()
    try:
        retrievers = {name: build_variant(spec, driver) for name, spec in specs.items()}
        summary, rankings = score_variants(benchmark, retrievers, split=args.split, top_k=args.top_k)
    finally:
        driver.close()
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split, "final_test": args.final_test, "top_k": args.top_k,
        "benchmark": args.benchmark, "benchmark_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
        "benchmark_corpus_paper_count": benchmark.get("corpus", {}).get("paper_count"),
        "live_corpus_sha256": hashlib.sha256(Path(CLEAN_FILE).read_bytes()).hexdigest(),
        "variants": specs,
        "label_source": "local Ollama AI-reference labels; not human ground truth",
        "note": "Live retrieval on the current stores scored against the frozen pool's labels; "
                "condensed metrics drop unjudged documents and must be read with the unjudged rate.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "rankings.jsonl").open("w") as stream:
        for row in rankings:
            stream.write(json.dumps(row) + "\n")
    for name, data in sorted(summary.items(), key=lambda item: -item[1]["condensed"]["ndcg@10"]):
        print(f"{name:26} condensed ndcg@10={data['condensed']['ndcg@10']:.4f} "
              f"standard ndcg@10={data['standard']['ndcg@10']:.4f} unjudged@10={data['unjudged_rate@10']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
