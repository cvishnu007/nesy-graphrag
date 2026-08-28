import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import (
    GRAPH_FUSION_WEIGHT,
    HOP_DEPTH,
    NEURAL_FUSION_WEIGHT,
    RRF_K,
    TOP_K,
)
from src.storage.chroma_store import query as chroma_query
from src.storage.neo4j_store import get_driver


def neural_retrieve(query, top_k=TOP_K):
    """Stage 1 — semantic search via SPECTER + ChromaDB."""
    return chroma_query(query, top_k=top_k)


def symbolic_expand(driver, paper_ids):
    """Stage 2 — 1-2 hop graph traversal via Neo4j CITES edges.

    NOTE: This intentionally does NOT exclude ``paper_ids`` from results
    so that papers found by both neural and symbolic retrieval can be
    correctly tagged as ``source="both"`` in ``nesy_retrieve()``.
    """
    hop_depth = max(1, int(HOP_DEPTH))
    query = f"""
        UNWIND $ids AS pid
        MATCH (p:Paper {{id: pid}})-[:CITES*1..{hop_depth}]-(related:Paper)
        WITH related, count(*) AS connections
        RETURN related.id       AS id,
               related.title    AS title,
               related.abstract AS abstract,
               related.year     AS year,
               related.category AS category,
               connections
        ORDER BY connections DESC, related.id
        LIMIT 20
    """
    with driver.session() as session:
        result = session.run(query, ids=paper_ids)

        records = list(result)
        max_connections = max((r["connections"] for r in records), default=1)
        expanded = []
        for rank, r in enumerate(records, 1):
            graph_score = r["connections"] / max_connections
            expanded.append({
                "id"       : r["id"],
                "title"    : r["title"],
                "abstract" : r["abstract"],
                "year"     : r["year"],
                "category" : r["category"],
                "score"    : round(graph_score, 6),
                "graph_score": round(graph_score, 6),
                "graph_connections": r["connections"],
                "graph_rank": rank,
                "source"   : "symbolic"
            })
        return expanded


def _rrf_score(rank, weight):
    """Return a rank contribution normalized to 1.0 for rank one."""
    rank_constant = max(0, RRF_K)
    return weight * (rank_constant + 1) / (rank_constant + rank)


def fuse_results(neural_papers, symbolic_papers, top_k):
    """Fuse neural and graph rankings using weighted reciprocal-rank fusion."""
    total_weight = NEURAL_FUSION_WEIGHT + GRAPH_FUSION_WEIGHT
    if total_weight <= 0:
        raise ValueError("At least one retrieval fusion weight must be positive")

    fused = {}

    for rank, paper in enumerate(neural_papers, 1):
        item = dict(paper)
        item["source"] = "neural"
        item["neural_rank"] = rank
        item["fusion_score"] = _rrf_score(rank, NEURAL_FUSION_WEIGHT)
        fused[item["id"]] = item

    for rank, paper in enumerate(symbolic_papers, 1):
        paper_id = paper["id"]
        contribution = _rrf_score(rank, GRAPH_FUSION_WEIGHT)
        if paper_id in fused:
            item = fused[paper_id]
            item["source"] = "both"
            item["graph_rank"] = rank
            item["graph_score"] = paper.get("graph_score", paper.get("score", 0.0))
            item["graph_connections"] = paper.get("graph_connections", 0)
            item["fusion_score"] += contribution
        else:
            item = dict(paper)
            item["source"] = "symbolic"
            item["graph_rank"] = rank
            item["fusion_score"] = contribution
            fused[paper_id] = item

    for item in fused.values():
        item["score"] = round(item.pop("fusion_score") / total_weight, 6)

    return sorted(
        fused.values(),
        key=lambda item: (-item["score"], item["id"]),
    )[:top_k]


def nesy_retrieve(driver, query, top_k=TOP_K):
    """Full NeSy retrieval — neural + symbolic combined and ranked."""
    neural_papers   = neural_retrieve(query, top_k)
    neural_ids      = [p["id"] for p in neural_papers]
    symbolic_papers = symbolic_expand(driver, neural_ids)

    return fuse_results(neural_papers, symbolic_papers, top_k)


def vector_only_retrieve(query, top_k=TOP_K):
    """Baseline retrieval — ChromaDB only, no symbolic expansion.

    Used by the baseline-comparison harness to measure the value
    added by the Neo4j graph layer.
    """
    papers = chroma_query(query, top_k=top_k)
    for p in papers:
        p["source"] = "neural"
    return papers
