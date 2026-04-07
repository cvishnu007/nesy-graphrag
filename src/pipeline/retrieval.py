import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import TOP_K
from src.storage.chroma_store import query as chroma_query
from src.storage.neo4j_store import get_driver


def neural_retrieve(query, top_k=TOP_K):
    """Stage 1 — semantic search via SPECTER + ChromaDB."""
    return chroma_query(query, top_k=top_k)


def symbolic_expand(driver, paper_ids):
    """Stage 2 — 1-2 hop graph traversal via Neo4j CITES edges."""
    with driver.session() as session:
        result = session.run("""
            UNWIND $ids AS pid
            MATCH (p:Paper {id: pid})-[:CITES*1..2]-(related:Paper)
            WHERE NOT related.id IN $ids
            WITH related, count(*) AS connections
            RETURN related.id       AS id,
                   related.title    AS title,
                   related.abstract AS abstract,
                   related.year     AS year,
                   related.category AS category,
                   connections
            ORDER BY connections DESC
            LIMIT 10
        """, ids=paper_ids)

        expanded = []
        for r in result:
            expanded.append({
                "id"       : r["id"],
                "title"    : r["title"],
                "abstract" : r["abstract"],
                "year"     : r["year"],
                "category" : r["category"],
                "score"    : r["connections"] * 0.1,
                "source"   : "symbolic"
            })
        return expanded


def nesy_retrieve(driver, query, top_k=TOP_K):
    """Full NeSy retrieval — neural + symbolic combined and ranked."""
    neural_papers   = neural_retrieve(query, top_k)
    neural_ids      = [p["id"] for p in neural_papers]
    symbolic_papers = symbolic_expand(driver, neural_ids)

    seen = {}
    for p in neural_papers:
        seen[p["id"]] = p
    for p in symbolic_papers:
        if p["id"] in seen:
            seen[p["id"]]["score"] += p["score"]
            seen[p["id"]]["source"] = "both"
        else:
            seen[p["id"]] = p

    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]
