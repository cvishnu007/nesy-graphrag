"""Graph-only retrieval using Neo4j concepts without Chroma seeds."""

import math
import re

from src.ingestion.ner_extractor import NOISE, filter_entities
from src.storage.neo4j_store import get_driver
from src.utils.config import (
    GRAPH_ONLY_CANDIDATE_LIMIT,
    TOP_K,
)


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
QUERY_STOPWORDS = NOISE | {
    "and",
    "or",
    "for",
    "from",
    "in",
    "of",
    "on",
    "to",
    "using",
    "versus",
    "vs",
    "with",
    "compared",
}


GRAPH_ONLY_QUERY = """
UNWIND $query_terms AS queryTerm
MATCH (concept:Concept)
WHERE concept.name = queryTerm
   OR (
       size(queryTerm) >= 4
       AND concept.name CONTAINS queryTerm
   )
   OR (
       size(concept.name) >= 4
       AND queryTerm CONTAINS concept.name
   )
WITH concept, collect(DISTINCT queryTerm) AS matchedTerms
ORDER BY size(matchedTerms) DESC, concept.name
LIMIT $candidate_limit

MATCH (paper:Paper)-[:RELATED_TO]->(concept)
UNWIND matchedTerms AS matchedTerm
WITH paper,
     collect(DISTINCT concept.name) AS matchedConcepts,
     collect(DISTINCT matchedTerm) AS matchedTerms
ORDER BY size(matchedTerms) DESC,
         size(matchedConcepts) DESC,
         coalesce(paper.citationCount, 0) DESC,
         paper.id
LIMIT $candidate_limit

OPTIONAL MATCH (paper)-[citation:CITES]-(:Paper)
RETURN paper.id AS id,
       paper.title AS title,
       paper.abstract AS abstract,
       paper.year AS year,
       paper.category AS category,
       matchedConcepts,
       matchedTerms,
       count(DISTINCT citation) AS citationDegree
"""


def normalize_query_concepts(query: str) -> list[str]:
    """Create stable one-to-four-word phrases from a query."""
    tokens = [
        token
        for token in TOKEN_PATTERN.findall(str(query).lower())
        if token not in QUERY_STOPWORDS
    ]

    phrases = []
    seen = set()

    for size in range(min(4, len(tokens)), 0, -1):
        for start in range(len(tokens) - size + 1):
            phrase = " ".join(
                tokens[start:start + size]
            )
            if phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)

    return filter_entities(phrases)


def _normalise_records(
    records,
    query_term_count: int,
) -> list[dict]:
    """Score and normalize Neo4j result records."""
    rows = [dict(record) for record in records]

    if not rows:
        return []

    max_concept_count = max(
        len(set(row.get("matchedConcepts") or []))
        for row in rows
    ) or 1
    max_citation_degree = max(
        int(row.get("citationDegree") or 0)
        for row in rows
    )

    papers = []

    for row in rows:
        matched_concepts = sorted(
            set(row.get("matchedConcepts") or [])
        )
        matched_terms = sorted(
            set(row.get("matchedTerms") or [])
        )
        citation_degree = int(
            row.get("citationDegree") or 0
        )

        query_coverage = (
            len(matched_terms) / query_term_count
            if query_term_count
            else 0.0
        )
        concept_strength = (
            len(matched_concepts) / max_concept_count
        )
        citation_strength = (
            math.log1p(citation_degree)
            / math.log1p(max_citation_degree)
            if max_citation_degree > 0
            else 0.0
        )

        score = (
            0.7 * query_coverage
            + 0.2 * concept_strength
            + 0.1 * citation_strength
        )

        papers.append(
            {
                "id": str(row.get("id", "")),
                "title": str(row.get("title", "") or ""),
                "abstract": str(
                    row.get("abstract", "") or ""
                ),
                "year": row.get("year", ""),
                "category": str(
                    row.get("category", "") or ""
                ),
                "score": round(score, 6),
                "graph_score": round(score, 6),
                "matched_concepts": matched_concepts,
                "matched_query_terms": matched_terms,
                "citation_degree": citation_degree,
                "source": "graph",
            }
        )

    return sorted(
        papers,
        key=lambda paper: (
            -paper["score"],
            paper["id"],
        ),
    )


def graph_only_retrieve(
    driver,
    query: str,
    top_k: int = TOP_K,
    candidate_limit: int = GRAPH_ONLY_CANDIDATE_LIMIT,
) -> list[dict]:
    """Retrieve papers only through query-to-concept graph matches."""
    top_k = int(top_k)
    candidate_limit = int(candidate_limit)

    if top_k <= 0:
        return []
    if candidate_limit <= 0:
        raise ValueError(
            "candidate_limit must be greater than zero"
        )
    if driver is None:
        raise ValueError("A Neo4j driver is required")

    query_terms = normalize_query_concepts(query)
    if not query_terms:
        return []

    with driver.session() as session:
        records = list(
            session.run(
                GRAPH_ONLY_QUERY,
                query_terms=query_terms,
                candidate_limit=candidate_limit,
            )
        )

    papers = _normalise_records(
        records,
        query_term_count=len(query_terms),
    )

    unique = []
    seen_ids = set()

    for paper in papers:
        if not paper["id"] or paper["id"] in seen_ids:
            continue
        seen_ids.add(paper["id"])
        unique.append(paper)

        if len(unique) >= top_k:
            break

    return unique


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Search Neo4j without Chroma"
    )
    parser.add_argument(
        "query",
        help="Research query",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
    )
    arguments = parser.parse_args()

    neo4j_driver = get_driver()

    try:
        for rank, paper in enumerate(
            graph_only_retrieve(
                neo4j_driver,
                arguments.query,
                top_k=arguments.top_k,
            ),
            start=1,
        ):
            concepts = ", ".join(
                paper["matched_concepts"][:3]
            )
            print(
                f"{rank:>2}. {paper['score']:.4f} "
                f"{paper['id']} {paper['title']} "
                f"[{concepts}]"
            )
    finally:
        neo4j_driver.close()