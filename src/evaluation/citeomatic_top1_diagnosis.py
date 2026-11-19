"""Top-1 diagnosis for RESULTS 2h: why co-citation lowers MRR on PubMed but not DBLP.

Validation split, first 600 queries per dataset. Usage:
    python src/evaluation/citeomatic_top1_diagnosis.py results/retrieval/citeomatic_20261119/top1_diagnosis_valid.json
"""
import json
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, ".")
from src.evaluation import citeomatic_benchmark as cb
from src.evaluation.citation_recommendation import cocitation_ranking, rrf
from src.evaluation.retrievers.bm25_retrieval import BM25Retriever
from src.storage.chroma_store import get_embedder

out = {}
for dataset in ("dblp", "pubmed"):
    papers = cb.load(dataset)
    train = set(cb.split_ids(papers, dataset, "train"))
    train_ids = sorted(train)
    queries = [(p, cb.gold(papers, p, train)) for p in cb.split_ids(papers, dataset, "valid")]
    queries = [(p, g) for p, g in queries if len(g) >= cb.MIN_GOLD][:600]
    indeg = Counter(c for p in train_ids for c in papers[p]["out"] & train)
    doc = np.load(cb.DATA / dataset / "train_bge.npy")
    qv = get_embedder().encode([cb.text(papers[p]) for p, _ in queries], normalize_embeddings=True).astype(np.float32)
    bm25 = BM25Retriever([{"id": p, "title": papers[p]["title"], "abstract": papers[p]["abstract"]} for p in train_ids])
    cites = {p: papers[p]["out"] & train for p in train_ids}
    stats = {k: [] for k in ("text_top1_indeg", "cocit_top1_indeg", "gold_indeg", "text_top1_hit", "cocit_top1_hit",
                             "graph_only_top1_hit", "gold_in_cocit_pool")}
    for i, (pid, g) in enumerate(queries):
        s = doc @ qv[i]
        order = np.argsort(-s)[:100]
        bge = [train_ids[j] for j in order]
        lex = [r["id"] for r in bm25.retrieve(cb.text(papers[pid]), top_k=100)]
        graph = cocitation_ranking(bge[:50], [s[j] for j in order[:50]], cites, lambda c: True)
        text = rrf([bge, lex], [1.0, 1.0])
        fused = rrf([bge, lex, graph], [1.0, 1.0, 1.0])
        stats["text_top1_indeg"].append(indeg[text[0]])
        stats["cocit_top1_indeg"].append(indeg[fused[0]])
        stats["gold_indeg"].append(float(np.median([indeg[x] for x in g])))
        stats["text_top1_hit"].append(text[0] in g)
        stats["cocit_top1_hit"].append(fused[0] in g)
        stats["graph_only_top1_hit"].append(bool(graph) and graph[0] in g)
        stats["gold_in_cocit_pool"].append(len(set(graph) & g) / len(g))
    out[dataset] = {k: round(float(np.median(v)) if "indeg" in k else float(np.mean(v)), 3) for k, v in stats.items()}
    out[dataset]["train_median_indeg"] = float(np.median([indeg[p] for p in train_ids]))
    out[dataset]["queries"] = len(queries)
    print(dataset, out[dataset], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=2)
