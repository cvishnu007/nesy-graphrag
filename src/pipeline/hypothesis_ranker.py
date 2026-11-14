"""Method-task hypothesis ranker: the generator form of the time-sliced benchmark scorer.

In the time-sliced benchmark (docs/RESULTS.md §4c), ranking (task, method) pairs
by RRF(vector, entity bridge, citation bridge) gave precision@10 0.067 for
predicting 2023-2025 combinations from pre-2023 data. That is 1.8x the
vector-neighbour baseline (0.038) and 3x the paper-candidate bridge generator
(0.021, §4d). This module applies the same scorer to the full corpus:

1. Seed papers = top 5 by BGE similarity to the query; target tasks = the specific
   Task entities they mention (vocabulary: document frequency 3..1% of papers,
   specific names, as in the benchmark).
2. For each target task T and each vocabulary Method M never co-occurring with T:
   * vector: cosine between the mean BGE vectors of T's papers and M's papers;
   * entity bridge: Adamic-Adar over entities co-occurring with both;
   * citation bridge: citation links between T's papers and M's papers.
3. RRF of the three scores per task; pairs are merged across target tasks and the
   top ``top_k`` are returned with their evidence (bridging entities, an example
   paper using M with a bridging entity, citation links).

Vocabulary also passes ``is_informative`` (no generic "X methods" heads, no ambiguous
task acronyms of 3 characters or fewer). In the benchmark this keeps a 1.9x precision@10
advantage over vector similarity (94 tasks; RESULTS §4d). Structures are built once per
process (about a minute) and cached.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def _model():
    from scipy.sparse import csr_matrix

    from src.evaluation.contradiction_index import PLAN, citation_map
    from src.evaluation.temporal_hypothesis import load_mentions
    from src.pipeline.hypothesis_bridge import is_informative, is_specific
    from src.storage.chroma_store import get_collection
    from src.utils.config import CLEAN_FILE

    corpus = json.loads(Path(CLEAN_FILE).read_text(encoding="utf-8"))
    titles = {p["id"]: p.get("clean_title", p.get("title", "")) for p in corpus}
    types, entities = defaultdict(Counter), defaultdict(set)
    for pid, name, kind in load_mentions():
        entities[pid].add(name)
        types[name][kind] += 1
    etype = {n: c.most_common(1)[0][0] for n, c in types.items()}
    papers = sorted(entities)
    df = Counter(e for pid in papers for e in entities[pid])
    vocab = sorted(e for e, n in df.items() if 3 <= n <= 0.01 * len(corpus) and is_specific(e)
                   and etype[e] in ("Task", "Method") and is_informative(e, etype[e]))
    index = {e: i for i, e in enumerate(vocab)}
    row = {pid: i for i, pid in enumerate(papers)}
    r, c = zip(*[(row[pid], index[e]) for pid in papers for e in entities[pid] if e in index])
    A = csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(len(papers), len(vocab)))
    co = (A.T @ A).tocsr()
    co.setdiag(0)
    co.eliminate_zeros()
    co_bin = co.copy()
    co_bin.data[:] = 1.0
    dfv = np.asarray(A.sum(axis=0)).ravel()
    cites = citation_map(corpus, json.loads(PLAN.read_text()))
    gr, gc = [], []
    for s, targets in cites.items():
        for t in targets:
            if s in row and t in row:
                gr += [row[s], row[t]]
                gc += [row[t], row[s]]
    G = csr_matrix((np.ones(len(gr), dtype=np.float32), (gr, gc)), shape=(len(papers),) * 2)
    G.data[:] = 1.0
    stored = get_collection().get(include=["embeddings"])  # all vectors; an id list this long exceeds SQLite limits
    emb = dict(zip(stored["ids"], stored["embeddings"]))
    E = np.stack([np.asarray(emb.get(pid, np.zeros(768)), dtype=np.float32) for pid in papers])
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    centroids = np.asarray((A.T @ E) / np.maximum(dfv[:, None], 1.0), dtype=np.float32)
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    methods = np.array([index[e] for e in vocab if etype[e] == "Method"])
    return {"vocab": vocab, "index": index, "etype": etype, "A": A.tocsc(), "co": co, "co_bin": co_bin,
            "dfv": dfv, "AG": (G @ A).tocsc(), "centroids": centroids, "methods": methods,
            "papers": papers, "entities": entities, "titles": titles,
            "aa": 1.0 / np.log(np.maximum(dfv, 2.0))}


def rank_methods_for_task(model, t):
    """RRF(vector, entity bridge, citation bridge) over Methods not co-occurring with task index t."""
    from src.evaluation.temporal_hypothesis import rrf_scores

    known = set(model["co"].indices[model["co"].indptr[t]:model["co"].indptr[t + 1]])
    candidates = np.array([m for m in model["methods"] if m not in known])
    vector = model["centroids"][candidates] @ model["centroids"][t]
    entity = np.asarray(model["co_bin"][candidates] @ (model["co_bin"][t].toarray().ravel() * model["aa"])).ravel()
    t_papers = model["A"][:, t].toarray().ravel()
    links = np.asarray(model["AG"][:, candidates].T @ t_papers).ravel()
    citation = links / np.sqrt(model["dfv"][t] * model["dfv"][candidates])
    return candidates, rrf_scores([vector, entity, citation]), links


def evidence(model, t, m, limit=3):
    """Bridging entities (co-occurring with both, rarest first) and one example paper for M."""
    row_t = set(model["co_bin"][t].indices)
    bridges = sorted(row_t & set(model["co_bin"][m].indices), key=lambda e: model["dfv"][e])[:limit]
    names = [model["vocab"][e] for e in bridges]
    method = model["vocab"][m]
    example = next((pid for pid in model["papers"] if method in model["entities"][pid]
                    and any(n in model["entities"][pid] for n in names)), None)
    return names, example


def rank_hypotheses(query, top_k=5, *, seeds=5):
    from src.pipeline.retrieval import neural_retrieve

    model = _model()
    seed_papers = neural_retrieve(query, top_k=seeds)
    task_counts = Counter(e for p in seed_papers for e in model["entities"].get(p["id"], ())
                          if e in model["index"] and model["etype"][e] == "Task")
    best = {}
    for task, support in task_counts.most_common(3):
        t = model["index"][task]
        candidates, scores, links = rank_methods_for_task(model, t)
        for position in np.argsort(-scores)[:top_k * 2]:
            m = int(candidates[position])
            key = model["vocab"][m]
            score = float(scores[position]) * (1 + 0.1 * (support - 1))
            if key not in best or best[key]["evidence_score"] < score:
                best[key] = {"task": task, "method": key, "t": t, "m": m, "evidence_score": round(score, 5),
                             "citation_links": int(links[position]), "task_support": support}
    ranked = sorted(best.values(), key=lambda h: (-h["evidence_score"], h["method"]))[:top_k]
    out = []
    for h in ranked:
        bridges, example = evidence(model, h.pop("t"), h.pop("m"))
        out.append({**h, "bridging_entities": bridges, "example_paper_id": example,
                    "example_paper_title": model["titles"].get(example, ""),
                    "seed_paper_ids": [p["id"] for p in seed_papers], "candidate_method": "method_task_ranker"})
    return out
