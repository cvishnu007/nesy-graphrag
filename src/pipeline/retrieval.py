import os
import re
import sys
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import (
    ALLOWED_EXPANSION_EDGES,
    BM25_FUSION_WEIGHT,
    GRAPH_CANDIDATE_LIMIT,
    GRAPH_EXPANSION_EDGES,
    GRAPH_FUSION_WEIGHT,
    GRAPH_GATE_MODE,
    GRAPH_HIGH_SEMANTIC_THRESHOLD,
    GRAPH_MIN_QUERY_TERM_COVERAGE,
    GRAPH_RELATIVE_MARGIN,
    GRAPH_SEMANTIC_FLOOR,
    GRAPH_STRONG_CONNECTIONS,
    HOP_DEPTH,
    NEURAL_FUSION_WEIGHT,
    RRF_K,
    TOP_K,
)
from src.storage.chroma_store import (
    query as chroma_query,
    score_papers_against_query,
)
from src.storage.neo4j_store import get_driver


def neural_retrieve(query, top_k=TOP_K):
    """Stage 1 — semantic search via SPECTER + ChromaDB."""
    return chroma_query(query, top_k=top_k)


def _expansion_pattern(edge_types):
    """Return a validated relationship-type alternation for Cypher."""
    edges = tuple(edge_types)
    if not edges or not set(edges) <= ALLOWED_EXPANSION_EDGES:
        raise ValueError(f"Expansion edges must be drawn from {sorted(ALLOWED_EXPANSION_EDGES)}")
    return "|".join(edges)


def symbolic_expand(driver, paper_ids, *, edge_types=None, limit=None):
    """Stage 2 — 1-2 hop graph traversal via Neo4j relationship edges.

    ``edge_types`` defaults to ``GRAPH_EXPANSION_EDGES`` (CITES only), and
    ``limit`` to ``GRAPH_CANDIDATE_LIMIT`` (20), reproducing the original query.

    NOTE: This intentionally does NOT exclude ``paper_ids`` from results
    so that papers found by both neural and symbolic retrieval can be
    correctly tagged as ``source="both"`` in ``nesy_retrieve()``.
    """
    hop_depth = max(1, int(HOP_DEPTH))
    pattern = _expansion_pattern(edge_types or GRAPH_EXPANSION_EDGES)
    candidate_limit = int(limit or GRAPH_CANDIDATE_LIMIT)
    if candidate_limit < 1:
        raise ValueError("limit must be positive")
    query = f"""
        UNWIND $ids AS pid
        MATCH (p:Paper {{id: pid}})-[:{pattern}*1..{hop_depth}]-(related:Paper)
        WHERE related.id <> pid
        WITH related, count(DISTINCT pid) AS connections
        RETURN related.id       AS id,
               related.title    AS title,
               related.abstract AS abstract,
               related.year     AS year,
               related.category AS category,
               connections
        ORDER BY connections DESC, related.id
        LIMIT $limit
    """
    with driver.session() as session:
        result = session.run(query, ids=paper_ids, limit=candidate_limit)

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


_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "of", "on", "or", "the", "to", "using", "with",
}


def _content_terms(text):
    """Extract stable lowercase query terms without connector words."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if token not in _QUERY_STOPWORDS
    }


def query_term_coverage(query, paper):
    """Return the fraction of meaningful query terms present in a paper."""
    query_terms = _content_terms(query)
    if not query_terms:
        return 0.0
    paper_terms = _content_terms(
        f"{paper.get('title', '')} {paper.get('abstract', '')}"
    )
    return len(query_terms & paper_terms) / len(query_terms)


def filter_symbolic_candidates(
    query,
    symbolic_papers,
    semantic_scores,
    *,
    mode=None,
    seed_scores=None,
    margin=None,
):
    """Remove graph neighbours that lack sufficient query relevance.

    ``absolute`` mode (default) keeps a candidate that is strongly similar to
    the query, or that has a reasonable semantic match, high query-term
    coverage, and links to many distinct neural seed papers.

    ``relative`` mode keeps a candidate whose similarity is at least the
    weakest neural seed's similarity minus ``margin``. Absolute thresholds sit
    inside the similarity band of the top neural results for most queries, so
    they reject almost every graph neighbour; the relative gate adapts to each
    query's band instead. Tune ``margin`` on development queries only.
    """
    mode = (mode or GRAPH_GATE_MODE).lower()
    if mode not in {"absolute", "relative"}:
        raise ValueError("mode must be absolute or relative")
    margin = GRAPH_RELATIVE_MARGIN if margin is None else float(margin)
    seed_values = [float(value) for value in (seed_scores or []) if value is not None]
    if mode == "relative" and not seed_values:
        return []
    relative_floor = min(seed_values) - margin if seed_values else None

    retained = []
    for paper in symbolic_papers:
        semantic_similarity = float(semantic_scores.get(paper["id"], 0.0))
        term_coverage = query_term_coverage(query, paper)
        connections = int(paper.get("graph_connections", 0))
        if mode == "relative":
            if semantic_similarity < relative_floor:
                continue
            reason = "within_relative_margin"
        else:
            semantic_match = semantic_similarity >= GRAPH_HIGH_SEMANTIC_THRESHOLD
            supported_match = (
                semantic_similarity >= GRAPH_SEMANTIC_FLOOR
                and term_coverage >= GRAPH_MIN_QUERY_TERM_COVERAGE
                and connections >= GRAPH_STRONG_CONNECTIONS
            )
            if not (semantic_match or supported_match):
                continue
            reason = (
                "high_semantic_similarity"
                if semantic_match
                else "strong_multi_seed_support"
            )

        item = dict(paper)
        item["semantic_similarity"] = round(semantic_similarity, 6)
        item["query_term_coverage"] = round(term_coverage, 6)
        item["graph_filter_reason"] = reason
        retained.append(item)
    return retained


def relevant_symbolic_expand(driver, query, paper_ids, seed_scores=None):
    """Expand through the graph and retain only query-relevant neighbours."""
    candidates = symbolic_expand(driver, paper_ids)
    scores = score_papers_against_query(
        query,
        [paper["id"] for paper in candidates],
    )
    return filter_symbolic_candidates(
        query, candidates, scores, seed_scores=seed_scores,
    )


_BM25 = None


def lexical_retrieve(query, top_k=TOP_K):
    """BM25 over the cleaned title+abstract corpus (lazily indexed once)."""
    global _BM25
    if _BM25 is None:
        from src.evaluation.retrievers.bm25_retrieval import BM25Retriever
        from src.utils.config import CLEAN_FILE

        _BM25 = BM25Retriever.from_json(CLEAN_FILE)
    return _BM25.retrieve(query, top_k=top_k)


def _rrf_score(rank, weight):
    """Return a rank contribution normalized to 1.0 for rank one."""
    rank_constant = max(0, RRF_K)
    return weight * (rank_constant + 1) / (rank_constant + rank)


def _add_lexical(fused, lexical_papers, weight):
    for rank, paper in enumerate(lexical_papers or [], 1):
        contribution = _rrf_score(rank, weight)
        item = fused.get(paper["id"])
        if item is None:
            item = dict(paper)
            item["source"] = "lexical"
            item["fusion_score"] = 0.0
            fused[paper["id"]] = item
        elif item["source"] != "lexical":
            item["lexical_support"] = True
        item["lexical_rank"] = rank
        item["fusion_score"] += contribution


def fuse_results(neural_papers, symbolic_papers, top_k, *, lexical_papers=None,
                 lexical_weight=None):
    """Fuse neural, graph, and optional BM25 rankings with weighted RRF."""
    lexical_weight = BM25_FUSION_WEIGHT if lexical_weight is None else float(lexical_weight)
    if lexical_weight < 0:
        raise ValueError("lexical_weight cannot be negative")
    use_lexical = lexical_weight > 0 and lexical_papers is not None
    total_weight = NEURAL_FUSION_WEIGHT + GRAPH_FUSION_WEIGHT + (lexical_weight if use_lexical else 0.0)
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
            for field in (
                "semantic_similarity",
                "query_term_coverage",
                "graph_filter_reason",
            ):
                if field in paper:
                    item[field] = paper[field]
            item["fusion_score"] += contribution
        else:
            item = dict(paper)
            item["source"] = "symbolic"
            item["graph_rank"] = rank
            item["fusion_score"] = contribution
            fused[paper_id] = item

    if use_lexical:
        _add_lexical(fused, lexical_papers, lexical_weight)

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
    symbolic_papers = relevant_symbolic_expand(
        driver, query, neural_ids,
        seed_scores=[p.get("score") for p in neural_papers],
    )
    lexical_papers = lexical_retrieve(query, top_k) if BM25_FUSION_WEIGHT > 0 else None

    return fuse_results(
        neural_papers, symbolic_papers, top_k, lexical_papers=lexical_papers,
    )

def _citation_degrees(driver, paper_ids):
    """Return undirected citation degree for every known paper ID."""
    cypher = """
        MATCH (paper:Paper)
        WHERE paper.id IN $ids
        OPTIONAL MATCH (paper)-[citation:CITES]-(:Paper)
        RETURN paper.id AS id, count(DISTINCT citation) AS degree
    """
    with driver.session() as session:
        return {
            record["id"]: record["degree"]
            for record in session.run(cypher, ids=list(set(paper_ids)))
        }


def build_retrieval_diagnostics(neural_papers, symbolic_papers, final_papers, degrees):
    """Build a compact, serializable explanation of hybrid ranking decisions."""
    final_ranks = {paper["id"]: rank for rank, paper in enumerate(final_papers, 1)}
    final_by_id = {paper["id"]: paper for paper in final_papers}
    candidates = {}

    for rank, paper in enumerate(neural_papers, 1):
        candidates[paper["id"]] = {
            "id": paper["id"],
            "title": paper.get("title", ""),
            "neural_rank": rank,
            "neural_score": paper.get("neural_score", paper.get("score", 0.0)),
            "graph_rank": None,
            "graph_connections": 0,
        }

    for rank, paper in enumerate(symbolic_papers, 1):
        row = candidates.setdefault(
            paper["id"],
            {
                "id": paper["id"],
                "title": paper.get("title", ""),
                "neural_rank": None,
                "neural_score": None,
            },
        )
        row["graph_rank"] = rank
        row["graph_connections"] = paper.get("graph_connections", 0)
        row["semantic_similarity"] = paper.get("semantic_similarity")
        row["query_term_coverage"] = paper.get("query_term_coverage")
        row["graph_filter_reason"] = paper.get("graph_filter_reason")

    rows = []
    for paper_id, row in candidates.items():
        final = final_by_id.get(paper_id)
        row["citation_degree"] = degrees.get(paper_id, 0)
        row["final_rank"] = final_ranks.get(paper_id)
        row["final_score"] = final.get("score") if final else None
        row["source"] = final.get("source") if final else "dropped"
        row["decision"] = (
            "kept in final top-k" if final else "dropped below final cutoff"
        )
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["final_rank"] is None,
            row["final_rank"] or 10**9,
            row["graph_rank"] or 10**9,
            row["neural_rank"] or 10**9,
        )
    )
    return {
        "source_distribution": dict(Counter(paper["source"] for paper in final_papers)),
        "candidates": rows,
    }


def diagnose_retrieval(driver, query, top_k=TOP_K):
    """Run each retrieval stage and explain the resulting candidate cutoff."""
    neural_papers = neural_retrieve(query, top_k)
    raw_symbolic_papers = symbolic_expand(
        driver,
        [paper["id"] for paper in neural_papers],
    )
    semantic_scores = score_papers_against_query(
        query,
        [paper["id"] for paper in raw_symbolic_papers],
    )
    symbolic_papers = filter_symbolic_candidates(
        query,
        raw_symbolic_papers,
        semantic_scores,
        seed_scores=[paper.get("score") for paper in neural_papers],
    )
    lexical_papers = lexical_retrieve(query, top_k) if BM25_FUSION_WEIGHT > 0 else None
    final_papers = fuse_results(
        neural_papers, symbolic_papers, top_k, lexical_papers=lexical_papers,
    )
    all_ids = [paper["id"] for paper in neural_papers + symbolic_papers]
    report = build_retrieval_diagnostics(
        neural_papers,
        symbolic_papers,
        final_papers,
        _citation_degrees(driver, all_ids),
    )
    report.update({
        "query": query,
        "top_k": top_k,
        "neural_candidates": len(neural_papers),
        "raw_symbolic_candidates": len(raw_symbolic_papers),
        "symbolic_candidates": len(symbolic_papers),
        "symbolic_candidates_filtered": (
            len(raw_symbolic_papers) - len(symbolic_papers)
        ),
    })
    return report


def print_retrieval_diagnostics(report):
    """Print a concise terminal report from diagnose_retrieval()."""
    print(f"\nRetrieval diagnostics: {report['query']}")
    print(f"Final sources: {report['source_distribution']}")
    print(
        "Graph filter: "
        f"{report['raw_symbolic_candidates']} candidates -> "
        f"{report['symbolic_candidates']} retained"
    )
    print("rank source    neural graph links degree semantic terms score    title")
    for row in report["candidates"]:
        final_rank = row["final_rank"] if row["final_rank"] is not None else "-"
        neural_rank = row["neural_rank"] if row["neural_rank"] is not None else "-"
        graph_rank = row["graph_rank"] if row["graph_rank"] is not None else "-"
        score = f"{row['final_score']:.4f}" if row["final_score"] is not None else "-"
        semantic = row.get("semantic_similarity")
        semantic = f"{semantic:.3f}" if semantic is not None else "-"
        coverage = row.get("query_term_coverage")
        coverage = f"{coverage:.2f}" if coverage is not None else "-"
        print(
            f"{str(final_rank):>4} {row['source']:<9} {str(neural_rank):>6} "
            f"{str(graph_rank):>5} {row['graph_connections']:>5} "
            f"{row['citation_degree']:>6} {semantic:>8} {coverage:>5} "
            f"{score:>7}  {row['title'][:70]}"
        )


def vector_only_retrieve(query, top_k=TOP_K):
    """Baseline retrieval — ChromaDB only, no symbolic expansion.

    Used by the baseline-comparison harness to measure the value
    added by the Neo4j graph layer.
    """
    papers = chroma_query(query, top_k=top_k)
    for p in papers:
        p["source"] = "neural"
    return papers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explain NeSy retrieval ranking")
    parser.add_argument("query", help="Research query to diagnose")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    neo4j_driver = get_driver()
    try:
        print_retrieval_diagnostics(
            diagnose_retrieval(neo4j_driver, args.query, top_k=args.top_k)
        )
    finally:
        neo4j_driver.close()
