"""Three-way lexical, semantic, and graph retrieval with weighted RRF."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.evaluation.config import (
    HYBRID_BM25_WEIGHT,
    HYBRID_GRAPH_WEIGHT,
    HYBRID_VECTOR_WEIGHT,
)
from src.evaluation.retrievers.bm25_retrieval import bm25_retrieve
from src.evaluation.retrievers.graph_only_retrieval import graph_only_retrieve
from src.pipeline.retrieval import vector_only_retrieve
from src.utils.config import (
    RRF_K,
    TOP_K,
)


METHOD_ORDER = ("bm25", "vector", "graph")


def weighted_rrf_fuse(
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    weights: Mapping[str, float],
    *,
    top_k: int,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """Fuse component rankings with deterministic weighted reciprocal rank."""
    if top_k <= 0:
        return []
    if rrf_k < 0:
        raise ValueError("rrf_k cannot be negative")
    missing = [method for method in METHOD_ORDER if method not in rankings]
    if missing:
        raise ValueError(f"Missing rankings: {', '.join(missing)}")
    total_weight = sum(float(weights.get(method, 0.0)) for method in METHOD_ORDER)
    if total_weight <= 0:
        raise ValueError("At least one fusion weight must be positive")

    fused: dict[str, dict] = {}
    for method in METHOD_ORDER:
        weight = float(weights.get(method, 0.0))
        if weight < 0:
            raise ValueError("Fusion weights cannot be negative")
        seen = set()
        for rank, paper in enumerate(rankings[method], start=1):
            paper_id = str(paper.get("id", "")).strip()
            if not paper_id:
                raise ValueError(f"{method} returned a blank paper ID")
            if paper_id in seen:
                raise ValueError(f"{method} returned duplicate paper ID {paper_id}")
            seen.add(paper_id)
            if paper_id not in fused:
                fused[paper_id] = {
                    **dict(paper),
                    "id": paper_id,
                    "component_ranks": {},
                    "fusion_score": 0.0,
                }
            item = fused[paper_id]
            item["component_ranks"][method] = rank
            item["fusion_score"] += weight * (rrf_k + 1) / (rrf_k + rank)

    results = []
    for item in fused.values():
        item["score"] = round(item.pop("fusion_score") / total_weight, 6)
        item["source"] = "hybrid:" + "+".join(item["component_ranks"])
        results.append(item)
    return sorted(results, key=lambda item: (-item["score"], item["id"]))[:top_k]


def tuned_hybrid_retrieve(driver, query: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve with the frozen dev-tuned BM25/vector/graph fusion."""
    if top_k <= 0:
        return []
    rankings = {
        "bm25": bm25_retrieve(query, top_k=top_k),
        "vector": vector_only_retrieve(query, top_k=top_k),
        "graph": graph_only_retrieve(driver, query, top_k=top_k),
    }
    weights = {
        "bm25": HYBRID_BM25_WEIGHT,
        "vector": HYBRID_VECTOR_WEIGHT,
        "graph": HYBRID_GRAPH_WEIGHT,
    }
    return weighted_rrf_fuse(rankings, weights, top_k=top_k, rrf_k=RRF_K)
