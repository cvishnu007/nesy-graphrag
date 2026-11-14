"""Time-sliced hypothesis benchmark (retrospective literature-based discovery).

A hypothesis is "method M could be applied to task T" for a (Task, Method) pair
of SciERC entities (``SciEntity`` via ``MENTIONS``) that never co-occur in any
paper published up to the cutoff year. It counts as confirmed if the pair
co-occurs in at least ``--min-future`` papers published after the cutoff.
Scorers see only pre-cutoff papers: their entities, BGE vectors, and citations
between pre-cutoff papers. Entity frequencies and the generic filter are
recomputed on pre-cutoff papers alone, so nothing from after the cutoff leaks in.

Scorers (predeclared 2026-11-15, nothing tuned):
* ``random``; ``popularity`` (pre-cutoff document frequency of M);
* ``vector``: cosine between the mean BGE vectors of T's and M's papers;
* ``entity_bridge``: Adamic-Adar over shared co-occurring entities (classic ABC);
* ``citation_bridge``: citation links between T's papers and M's papers,
  cosine-normalised by paper counts;
* ``fusion_graph``: RRF(entity_bridge, citation_bridge);
* ``fusion_vector_graph``: RRF(vector, entity_bridge, citation_bridge);
* ``fusion_all``: RRF(popularity, vector, entity_bridge, citation_bridge).

For each eligible task, the candidates are every vocabulary Method not co-occurring
with it before the cutoff. Metrics per task: average precision, precision@20,
recall@100, and AUROC. Means over tasks; paired sign-flip tests across tasks.
Tasks are split 50/50 by hash into dev and test; both are reported (nothing is
selected on dev).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path("results/reasoning/temporal_hypothesis_20261115")


def rank_scores(scores):
    """Higher score -> rank 1; ties share the mean rank. Returns ranks (1-based)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    unique, inverse = np.unique(-scores, return_inverse=True)
    sums = np.bincount(inverse, weights=ranks)
    counts = np.bincount(inverse)
    return (sums / counts)[inverse]


def rrf_scores(score_lists, k=60):
    return sum(1.0 / (k + rank_scores(s)) for s in score_lists)


def task_metrics(scores, positives):
    ranks = rank_scores(scores)
    pos_ranks = np.sort(ranks[positives])
    n, p = len(scores), len(pos_ranks)
    ap = float(np.mean([(i + 1) / r for i, r in enumerate(pos_ranks)]))
    auroc = float((n - p) and (np.sum(n - pos_ranks) - p * (p - 1) / 2) / (p * (n - p)))
    return {"ap": ap, "p@20": float(np.sum(pos_ranks <= 20) / 20), "r@100": float(np.sum(pos_ranks <= 100) / p),
            "auroc": auroc}


def load_mentions():
    from src.storage.neo4j_store import get_driver

    driver = get_driver()
    try:
        with driver.session() as session:
            return [(r["pid"], r["name"], r["type"]) for r in session.run(
                "MATCH (p:Paper)-[:MENTIONS]->(e:SciEntity) RETURN p.id AS pid, e.name AS name, e.type AS type")]
    finally:
        driver.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cutoff", type=int, default=2022)
    parser.add_argument("--min-future", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=3, help="pre-cutoff papers an entity needs to be in the vocabulary")
    parser.add_argument("--max-df-fraction", type=float, default=0.01, help="pre-cutoff generic filter")
    parser.add_argument("--min-task-df", type=int, default=10)
    parser.add_argument("--min-task-positives", type=int, default=3)
    args = parser.parse_args(argv)
    from scipy.sparse import csr_matrix

    from src.evaluation.contradiction_index import PLAN, citation_map
    from src.evaluation.significance import paired_permutation_test
    from src.pipeline.hypothesis_bridge import is_specific
    from src.storage.chroma_store import get_collection
    from src.utils.config import CLEAN_FILE

    corpus = json.loads(Path(CLEAN_FILE).read_text(encoding="utf-8"))
    year = {p["id"]: int(p.get("year") or 0) for p in corpus}
    past = {pid for pid, y in year.items() if 0 < y <= args.cutoff}
    future = {pid for pid, y in year.items() if y > args.cutoff}
    mentions = load_mentions()
    types = defaultdict(Counter)
    paper_entities = defaultdict(set)
    for pid, name, kind in mentions:
        paper_entities[pid].add(name)
        types[name][kind] += 1
    etype = {name: counts.most_common(1)[0][0] for name, counts in types.items()}
    df_past = Counter(e for pid in past for e in paper_entities.get(pid, ()))
    max_df = args.max_df_fraction * len(past)
    vocab = sorted(e for e, n in df_past.items()
                   if args.min_df <= n <= max_df and is_specific(e) and etype[e] in ("Task", "Method"))
    index = {e: i for i, e in enumerate(vocab)}
    tasks_all = [e for e in vocab if etype[e] == "Task"]
    methods = np.array([index[e] for e in vocab if etype[e] == "Method"])
    print(f"past papers {len(past)}, future {len(future)}, vocabulary {len(vocab)} "
          f"({len(tasks_all)} tasks, {len(methods)} methods)", flush=True)

    past_list = sorted(past)
    row_of = {pid: i for i, pid in enumerate(past_list)}
    rows, cols = [], []
    for pid in past_list:
        for e in paper_entities.get(pid, ()):
            if e in index:
                rows.append(row_of[pid])
                cols.append(index[e])
    A = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(past_list), len(vocab)))
    co = (A.T @ A).tocsr()  # entity co-occurrence counts, pre-cutoff
    co.setdiag(0)
    co.eliminate_zeros()
    co_bin = co.copy()
    co_bin.data[:] = 1.0
    df_vec = np.asarray(A.sum(axis=0)).ravel()
    aa_weight = 1.0 / np.log(np.maximum(df_vec, 2.0))

    cites = citation_map(corpus, json.loads(PLAN.read_text()))
    g_rows, g_cols = [], []
    for source, targets in cites.items():
        if source in row_of:
            for target in targets:
                if target in row_of:
                    g_rows += [row_of[source], row_of[target]]
                    g_cols += [row_of[target], row_of[source]]
    G = csr_matrix((np.ones(len(g_rows), dtype=np.float32), (g_rows, g_cols)), shape=(len(past_list),) * 2)
    G.data[:] = 1.0
    print(f"pre-cutoff citation links (undirected): {G.nnz // 2}", flush=True)

    stored = get_collection().get(ids=past_list, include=["embeddings"])
    emb = {pid: np.asarray(v, dtype=np.float32) for pid, v in zip(stored["ids"], stored["embeddings"])}
    E = np.stack([emb.get(pid, np.zeros(768, dtype=np.float32)) for pid in past_list])
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    centroids = np.asarray((A.T @ E) / np.maximum(df_vec[:, None], 1.0), dtype=np.float32)
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)

    future_pairs = Counter()
    for pid in future:
        ents = [e for e in paper_entities.get(pid, ()) if e in index]
        t_ents = [e for e in ents if etype[e] == "Task"]
        m_ents = [e for e in ents if etype[e] == "Method"]
        for t in t_ents:
            for m in m_ents:
                future_pairs[(index[t], index[m])] += 1

    AG = (G @ A).tocsc()  # papers x entities: citation-neighbour mentions
    rng = np.random.default_rng(20261115)
    per_task = defaultdict(dict)
    eligible = []
    for task in tasks_all:
        t = index[task]
        if df_vec[t] < args.min_task_df:
            continue
        known = set(co.indices[co.indptr[t]:co.indptr[t + 1]])
        candidates = np.array([m for m in methods if m not in known])
        positives = np.array([future_pairs.get((t, m), 0) >= args.min_future for m in candidates])
        if positives.sum() < args.min_task_positives:
            continue
        eligible.append(task)
        t_neighbours = co_bin[t].toarray().ravel() * aa_weight
        entity_bridge = np.asarray(co_bin[candidates] @ t_neighbours).ravel()
        t_papers = A[:, t].toarray().ravel()
        links = np.asarray(AG[:, candidates].T @ t_papers).ravel()
        citation_bridge = links / np.sqrt(df_vec[t] * df_vec[candidates])
        scores = {
            "random": rng.random(len(candidates)),
            "popularity": df_vec[candidates].astype(float),
            "vector": centroids[candidates] @ centroids[t],
            "entity_bridge": entity_bridge,
            "citation_bridge": citation_bridge,
        }
        scores["fusion_graph"] = rrf_scores([scores["entity_bridge"], scores["citation_bridge"]])
        scores["fusion_vector_graph"] = rrf_scores([scores["vector"], scores["entity_bridge"], scores["citation_bridge"]])
        scores["fusion_all"] = rrf_scores([scores["popularity"], scores["vector"], scores["entity_bridge"],
                                           scores["citation_bridge"]])
        for name, values in scores.items():
            per_task[name][task] = {**task_metrics(np.asarray(values, dtype=float), positives),
                                    "candidates": int(len(candidates)), "positives": int(positives.sum())}

    split = {t: ("dev" if int(hashlib.sha256(f"lbd|{t}".encode()).hexdigest(), 16) % 2 == 0 else "test")
             for t in eligible}
    report = {"settings": vars(args), "past_papers": len(past), "future_papers": len(future),
              "vocabulary": len(vocab), "eligible_tasks": len(eligible),
              "mean_candidates": round(float(np.mean([per_task["random"][t]["candidates"] for t in eligible])), 1),
              "mean_positives": round(float(np.mean([per_task["random"][t]["positives"] for t in eligible])), 1),
              "splits": {}}
    comparisons = [("fusion_vector_graph", "vector"), ("fusion_graph", "vector"), ("entity_bridge", "vector"),
                   ("citation_bridge", "vector"), ("fusion_all", "popularity"), ("fusion_all", "fusion_vector_graph"),
                   ("vector", "random"), ("popularity", "random")]
    for part in ("all", "dev", "test"):
        chosen = [t for t in eligible if part == "all" or split[t] == part]
        summary = {name: {m: round(float(np.mean([values[t][m] for t in chosen])), 4)
                          for m in ("ap", "p@20", "r@100", "auroc")} for name, values in per_task.items()}
        tests = {}
        for a, b in comparisons:
            for metric in ("ap", "r@100"):
                deltas = [per_task[a][t][metric] - per_task[b][t][metric] for t in chosen]
                result = paired_permutation_test(deltas)
                tests[f"{a}_vs_{b}:{metric}"] = {"mean_delta": round(result["mean_delta"], 4),
                                                 "bootstrap_95_ci": [round(x, 4) for x in result["bootstrap_95_ci"]],
                                                 "p": result["monte_carlo_two_sided_p"]}
        report["splits"][part] = {"tasks": len(chosen), "scorers": summary, "comparisons": tests}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "splits"}, indent=1))
    for part in ("all", "test"):
        print(f"\n{part} ({report['splits'][part]['tasks']} tasks)")
        for name, m in sorted(report["splits"][part]["scorers"].items(), key=lambda item: -item[1]["ap"]):
            print(f"  {name:22} AP={m['ap']:.4f} P@20={m['p@20']:.4f} R@100={m['r@100']:.4f} AUROC={m['auroc']:.4f}")
        for name, r in report["splits"][part]["comparisons"].items():
            print(f"  {name:45} delta={r['mean_delta']:+.4f} CI={r['bootstrap_95_ci']} p={r['p']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
