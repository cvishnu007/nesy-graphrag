import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.pipeline.prompts import build_contradiction_prompt
from src.pipeline.retrieval import nesy_retrieve
from src.pipeline.verdicts import parse_contradiction_response
from src.utils.config import (
    CONTRADICTION_CANDIDATE_POOL,
    CONTRADICTION_MIN_CONFIDENCE,
    CONTRADICTION_MIN_CONCEPT_JACCARD,
    CONTRADICTION_MIN_SHARED_CONCEPTS,
    GROQ_MAX_RETRIES,
    LLM_MODEL,
    LLM_MODEL_FALLBACK,
)
from src.utils.groq_client import groq_chat_with_retry


def score_contradiction_candidate(shared_count, concepts1, concepts2, year1, year2):
    """Return normalized overlap, year gap, and deterministic candidate score."""
    union_count = concepts1 + concepts2 - shared_count
    concept_jaccard = shared_count / union_count if union_count > 0 else 0.0
    year_gap = abs(int(year1) - int(year2))
    candidate_score = 0.75 * concept_jaccard + 0.25 * min(year_gap / 5, 1.0)
    return concept_jaccard, year_gap, candidate_score


def detect_contradictions(driver, query, top_k=5):
    """Rank cross-year paper pairs by normalized concept overlap."""
    papers = nesy_retrieve(driver, query, top_k=10)
    paper_ids = [paper["id"] for paper in papers if paper.get("id")]
    if len(paper_ids) < 2:
        return []

    cypher = """
        UNWIND $ids AS id1
        UNWIND $ids AS id2
        WITH id1, id2
        WHERE id1 < id2
        MATCH (p1:Paper {id: id1})-[:RELATED_TO]->(shared:Concept)<-[:RELATED_TO]-(p2:Paper {id: id2})
        WITH p1, p2, collect(DISTINCT shared.name) AS shared_names
        WHERE size(shared_names) >= $min_shared AND p1.year <> p2.year
        OPTIONAL MATCH (p1)-[:RELATED_TO]->(c1:Concept)
        WITH p1, p2, shared_names, count(DISTINCT c1) AS concepts1
        OPTIONAL MATCH (p2)-[:RELATED_TO]->(c2:Concept)
        WITH p1, p2, shared_names, concepts1, count(DISTINCT c2) AS concepts2
        RETURN p1.id AS id1, p1.title AS title1, p1.abstract AS abstract1, p1.year AS year1,
               p2.id AS id2, p2.title AS title2, p2.abstract AS abstract2, p2.year AS year2,
               shared_names, concepts1, concepts2
        ORDER BY size(shared_names) DESC
        LIMIT $candidate_pool
    """
    with driver.session() as session:
        records = list(session.run(
            cypher,
            ids=paper_ids,
            min_shared=CONTRADICTION_MIN_SHARED_CONCEPTS,
            candidate_pool=max(top_k, CONTRADICTION_CANDIDATE_POOL),
        ))

    candidates = []
    for record in records:
        shared_count = len(record["shared_names"])
        concept_jaccard, year_gap, candidate_score = score_contradiction_candidate(
            shared_count,
            record["concepts1"],
            record["concepts2"],
            record["year1"],
            record["year2"],
        )
        if concept_jaccard < CONTRADICTION_MIN_CONCEPT_JACCARD:
            continue
        candidates.append({
            "paper1": {
                "id": record["id1"],
                "title": record["title1"],
                "abstract": record["abstract1"],
                "year": record["year1"],
            },
            "paper2": {
                "id": record["id2"],
                "title": record["title2"],
                "abstract": record["abstract2"],
                "year": record["year2"],
            },
            "shared_concepts": shared_count,
            "shared_concept_names": record["shared_names"],
            "concept_jaccard": round(concept_jaccard, 4),
            "year_gap": year_gap,
            "candidate_score": round(candidate_score, 4),
            "flag": (
                "Cross-year pair with normalized concept overlap "
                f"{concept_jaccard:.2f}"
            ),
        })

    return sorted(
        candidates,
        key=lambda pair: (-pair["candidate_score"], -pair["shared_concepts"]),
    )[:top_k]


def llm_contradict(groq_client, driver, query, top_k=5):
    print(f"\n{'=' * 60}")
    print(f"[CONTRADICTION MODE] {query}")
    print(f"{'=' * 60}")

    candidates = detect_contradictions(driver, query, top_k=top_k)
    print(f"[Graph] Found {len(candidates)} candidate pairs")
    if not candidates:
        print("No contradiction candidates found.")
        return {"query": query, "contradictions": []}

    checked_pairs = []
    for index, pair in enumerate(candidates, 1):
        paper1 = pair["paper1"]
        paper2 = pair["paper2"]
        prompt = build_contradiction_prompt(paper1, paper2)
        print(f"[LLM] Checking pair {index}/{len(candidates)}...")

        try:
            analysis = groq_chat_with_retry(
                groq_client,
                prompt,
                model=LLM_MODEL,
                fallback_model=LLM_MODEL_FALLBACK,
                max_tokens=300,
                temperature=0.0,
                max_retries=GROQ_MAX_RETRIES,
            )
        except Exception as error:
            analysis = f"LLM call failed: {error}"

        parsed = parse_contradiction_response(analysis)
        checked_pairs.append({
            "paper1": paper1,
            "paper2": paper2,
            "shared_concepts": pair["shared_concepts"],
            "shared_concept_names": pair["shared_concept_names"],
            "concept_jaccard": pair["concept_jaccard"],
            "candidate_score": pair["candidate_score"],
            "llm_analysis": analysis,
            "verdict": parsed["verdict"],
            "confidence": parsed["confidence"],
            "verdict_valid": parsed["valid"],
            "accepted_contradiction": (
                parsed["verdict"] == "CONTRADICTION"
                and parsed["confidence"] is not None
                and parsed["confidence"] >= CONTRADICTION_MIN_CONFIDENCE
            ),
        })

        print(f"\n  Pair {index}:")
        print(f"  P1: {paper1['title'][:60]}... ({paper1['year']})")
        print(f"  P2: {paper2['title'][:60]}... ({paper2['year']})")
        print(f"  Verdict: {parsed['verdict']} (confidence={parsed['confidence']})")

    return {"query": query, "contradictions": checked_pairs}
