"""Citation recommendation: does the citation graph help find the papers a paper cites?

Task: given a corpus paper's title and abstract, rank the corpus so that the
papers it actually cites come first. Ground truth comes from real reference
lists (original ``CITES`` plus fetched ``CITES_ADDED``), so no LLM judge is involved.

Leakage controls:
* the query paper's own citation edges are never used; the graph signal comes
  only from *other* papers' reference lists;
* papers that cite the query paper are excluded from the neighbour set;
* every method ranks only papers published no later than the query paper, and
  never the query paper itself.

Methods: BGE (stored document vectors, cosine), BM25 (title + abstract as the
query), RRF(BGE, BM25), co-citation graph (papers cited by the query's BGE
neighbours, weighted by neighbour similarity), and RRF(BGE, BM25, graph).

Protocol (predeclared 2026-11-15): 600 query papers (year >= 2022, >= 5 cited
corpus papers no newer than themselves) are split 200 dev / 400 test by hash.
Graph settings (neighbours k in {10, 20, 50}, fusion weight w in {0.5, 1.0})
are chosen on dev by Recall@20; the chosen setting runs once on test. Paired
Monte Carlo sign-flip tests with bootstrap CIs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("results/retrieval/citation_recommendation_20261115")
K_VALUES = (5, 10, 20)
GRID = [(k, w) for k in (10, 20, 50) for w in (0.5, 1.0)]
DEPTH = 100


def rrf(rankings, weights, depth=DEPTH, k=60):
    scores = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, paper_id in enumerate(ranking, 1):
            scores[paper_id] += weight / (k + rank)
    return [p for p, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))][:depth]


def cocitation_ranking(neighbours, similarities, cites, allowed, depth=DEPTH):
    scores = defaultdict(float)
    for neighbour, similarity in zip(neighbours, similarities):
        for cited in cites.get(neighbour, ()):
            if allowed(cited):
                scores[cited] += float(similarity)
    return [p for p, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))][:depth]


def select_queries(corpus, cites, *, count=600, seed="citrec-v1"):
    year = {p["id"]: int(p.get("year") or 0) for p in corpus}
    eligible = []
    for paper in corpus:
        pid = paper["id"]
        targets = {t for t in cites.get(pid, ()) if t != pid and 0 < year.get(t, 0) <= year[pid]}
        if year[pid] >= 2022 and len(targets) >= 5:
            eligible.append((pid, targets))
    eligible.sort(key=lambda item: hashlib.sha256(f"{seed}|{item[0]}".encode()).hexdigest())
    chosen = eligible[:count]
    return chosen[:count // 3], chosen[count // 3:], len(eligible)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args(argv)
    from src.evaluation.contradiction_index import PLAN, citation_map
    from src.evaluation.ir_metrics import aggregate_query_metrics, evaluate_ranking
    from src.evaluation.retrievers.bm25_retrieval import BM25Retriever
    from src.evaluation.significance import paired_permutation_test
    from src.storage.chroma_store import get_collection
    from src.utils.config import CLEAN_FILE

    corpus = json.loads(Path(CLEAN_FILE).read_text(encoding="utf-8"))
    year = {p["id"]: int(p.get("year") or 0) for p in corpus}
    cites = citation_map(corpus, json.loads(PLAN.read_text()))
    cited_by = defaultdict(set)
    for source, targets in cites.items():
        for target in targets:
            cited_by[target].add(source)
    dev, test, eligible = select_queries(corpus, cites)
    OUT.mkdir(parents=True, exist_ok=True)

    stored = get_collection().get(include=["embeddings"])
    ids = list(stored["ids"])
    matrix = np.asarray(stored["embeddings"], dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    position = {pid: i for i, pid in enumerate(ids)}
    id_array = np.array(ids)
    years = np.array([year.get(pid, 0) for pid in ids])
    bm25 = BM25Retriever.from_json(CLEAN_FILE)
    text = {p["id"]: f"{p.get('clean_title', '')} {p.get('clean_abstract', '')}" for p in corpus}

    def base_rankings(pid):
        own = year[pid]
        sims = matrix @ matrix[position[pid]]
        sims[(years > own) | (years == 0)] = -np.inf
        sims[position[pid]] = -np.inf
        order = np.argsort(-sims)[:max(DEPTH, 60)]
        bge = [str(id_array[i]) for i in order[:DEPTH]]
        neighbours = [(str(id_array[i]), float(sims[i])) for i in order if str(id_array[i]) not in cited_by[pid]]
        lexical = [r["id"] for r in bm25.retrieve(text[pid], top_k=DEPTH * 3)
                   if r["id"] != pid and 0 < year.get(r["id"], 0) <= own][:DEPTH]
        return bge, lexical, neighbours

    def run(queries, settings):
        rows = defaultdict(list)
        for pid, targets in queries:
            bge, lexical, neighbours = base_rankings(pid)
            allowed = lambda c, own=year[pid], pid=pid: c != pid and 0 < year.get(c, 0) <= own
            judgments = {t: 1 for t in targets}
            lists = {"bge": bge, "bm25": lexical, "rrf_bge_bm25": rrf([bge, lexical], [1.0, 1.0])}
            for k, w in settings:
                top = neighbours[:k]
                graph = cocitation_ranking([n for n, _ in top], [s for _, s in top], cites, allowed)
                lists[f"graph_k{k}"] = graph
                lists[f"rrf_bge_bm25_graph_k{k}_w{w}"] = rrf([bge, lexical, graph], [1.0, 1.0, w])
            for name, ranking in lists.items():
                rows[name].append({"query_id": pid, **evaluate_ranking(ranking, judgments, K_VALUES)})
        summary = {}
        for name, values in rows.items():
            metrics = aggregate_query_metrics(values)["metrics"]
            summary[name] = {"queries": len(values),
                             **{m: round(metrics[m]["mean"], 4) for m in ("ndcg@10", "recall@20", "precision@10", "map", "mrr")},
                             "per_query_recall@20": {v["query_id"]: v["recall@20"] for v in values},
                             "per_query_ndcg@10": {v["query_id"]: v["ndcg@10"] for v in values}}
        return summary

    def compare(summary, pairs):
        out = {}
        for challenger, reference in pairs:
            for metric in ("recall@20", "ndcg@10"):
                a, b = summary[challenger][f"per_query_{metric}"], summary[reference][f"per_query_{metric}"]
                deltas = [a[q] - b[q] for q in sorted(a)]
                test_result = paired_permutation_test(deltas)
                out[f"{challenger}_vs_{reference}:{metric}"] = {
                    "mean_delta": round(test_result["mean_delta"], 4),
                    "bootstrap_95_ci": [round(x, 4) for x in test_result["bootstrap_95_ci"]],
                    "p": test_result["monte_carlo_two_sided_p"]}
        return out

    def show(summary, comparisons):
        for name, data in sorted(summary.items(), key=lambda item: -item[1]["recall@20"]):
            print(f"  {name:30} recall@20={data['recall@20']:.3f} ndcg@10={data['ndcg@10']:.3f} map={data['map']:.3f}")
        for name, data in comparisons.items():
            print(f"  {name:55} delta={data['mean_delta']:+.4f} CI={data['bootstrap_95_ci']} p={data['p']:.4f}")

    strip = lambda s: {n: {k: v for k, v in d.items() if not k.startswith("per_query")} for n, d in s.items()}
    selection_path = OUT / "selection.json"
    if not args.final_test:
        summary = run(dev, GRID)
        fused = {f"rrf_bge_bm25_graph_k{k}_w{w}": (k, w) for k, w in GRID}
        best = max(sorted(fused), key=lambda name: summary[name]["recall@20"])
        comparisons = compare(summary, [(best, "rrf_bge_bm25"), (best, "bge"), ("rrf_bge_bm25", "bge")])
        selection_path.write_text(json.dumps({"selected": best, "k": fused[best][0], "w": fused[best][1],
                                              "eligible_queries": eligible, "dev_queries": len(dev),
                                              "test_queries": len(test)}, indent=2) + "\n")
        (OUT / "dev_summary.json").write_text(json.dumps({"variants": strip(summary), "comparisons": comparisons},
                                                          indent=2) + "\n")
        print(f"eligible={eligible} dev={len(dev)} test={len(test)} selected={best}")
        show(strip(summary), comparisons)
        return 0
    selection = json.loads(selection_path.read_text())
    summary = run(test, [(selection["k"], selection["w"])])
    best = selection["selected"]
    comparisons = compare(summary, [(best, "rrf_bge_bm25"), (best, "bge"), (f"graph_k{selection['k']}", "bge"),
                                    ("rrf_bge_bm25", "bge")])
    (OUT / "test_summary.json").write_text(json.dumps({"selection": selection, "variants": strip(summary),
                                                       "comparisons": comparisons}, indent=2) + "\n")
    print(f"test ({len(test)} papers) selected={best}")
    show(strip(summary), comparisons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
