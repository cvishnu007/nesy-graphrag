import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.pipeline.prompts import build_contradiction_prompt
from src.pipeline.retrieval import nesy_retrieve
from src.pipeline.concepts import concept_cypher
from src.pipeline.verdicts import CONTRADICTION_JSON_SCHEMA, parse_contradiction_response
from src.utils.config import (
    CONCEPT_EXCLUDE_GENERIC,
    CONTRADICTION_CANDIDATE_POOL,
    CONTRADICTION_MIN_CONFIDENCE,
    CONTRADICTION_MIN_CONCEPT_JACCARD,
    CONTRADICTION_MIN_SHARED_CONCEPTS,
    CONTRADICTION_CANDIDATE_MODE,
    CONTRADICTION_REQUIRE_DIFFERENT_YEAR,
    CONTRADICTION_STRUCTURED_OUTPUT,
    CONTRADICTION_YEAR_WEIGHT,
)


def score_contradiction_candidate(shared_count, concepts1, concepts2, year1, year2,
                                  *, year_weight=None):
    """Return normalized overlap, year gap, and deterministic candidate score.

    ``year_weight`` (default ``CONTRADICTION_YEAR_WEIGHT`` = 0.25) blends a
    capped year gap into the score. It is a historical screening heuristic,
    not evidence of disagreement; set it to 0 to rank by overlap alone.
    """
    weight = CONTRADICTION_YEAR_WEIGHT if year_weight is None else float(year_weight)
    if not 0 <= weight <= 1:
        raise ValueError("year_weight must be in [0, 1]")
    union_count = concepts1 + concepts2 - shared_count
    concept_jaccard = shared_count / union_count if union_count > 0 else 0.0
    year_gap = abs(int(year1) - int(year2))
    candidate_score = (1 - weight) * concept_jaccard + weight * min(year_gap / 5, 1.0)
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
        WHERE NOT ($exclude_generic AND coalesce(shared.generic, false))
        WITH p1, p2, collect(DISTINCT shared.name) AS shared_names
        WHERE size(shared_names) >= $min_shared
          AND (NOT $require_different_year OR p1.year <> p2.year)
        OPTIONAL MATCH (p1)-[:RELATED_TO]->(c1:Concept)
        WHERE NOT ($exclude_generic AND coalesce(c1.generic, false))
        WITH p1, p2, shared_names, count(DISTINCT c1) AS concepts1
        OPTIONAL MATCH (p2)-[:RELATED_TO]->(c2:Concept)
        WHERE NOT ($exclude_generic AND coalesce(c2.generic, false))
        WITH p1, p2, shared_names, concepts1, count(DISTINCT c2) AS concepts2
        RETURN p1.id AS id1, p1.title AS title1, p1.abstract AS abstract1, p1.year AS year1,
               p2.id AS id2, p2.title AS title2, p2.abstract AS abstract2, p2.year AS year2,
               shared_names, concepts1, concepts2
        ORDER BY size(shared_names) DESC
        LIMIT $candidate_pool
    """
    with driver.session() as session:
        records = list(session.run(
            concept_cypher(cypher),
            ids=paper_ids,
            min_shared=CONTRADICTION_MIN_SHARED_CONCEPTS,
            candidate_pool=max(top_k, CONTRADICTION_CANDIDATE_POOL),
            exclude_generic=CONCEPT_EXCLUDE_GENERIC,
            require_different_year=CONTRADICTION_REQUIRE_DIFFERENT_YEAR,
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


def llm_contradict(llm_client, driver, query, top_k=5):
    print(f"\n{'=' * 60}")
    print(f"[CONTRADICTION MODE] {query}")
    print(f"{'=' * 60}")

    if CONTRADICTION_CANDIDATE_MODE == "citation":
        from src.pipeline.contradiction_citation import detect_contradictions_from_index

        candidates = detect_contradictions_from_index(driver, query, top_k=top_k)
    elif CONTRADICTION_CANDIDATE_MODE == "nli":
        from src.pipeline.contradiction_nli import detect_contradictions_nli

        candidates = detect_contradictions_nli(driver, query, top_k=top_k)
    else:
        candidates = detect_contradictions(driver, query, top_k=top_k)
    print(f"[Graph] Found {len(candidates)} candidate pairs")
    if not candidates:
        print("No contradiction candidates found.")
        return {"query": query, "contradictions": []}

    checked_pairs = []
    for index, pair in enumerate(candidates, 1):
        paper1 = pair["paper1"]
        paper2 = pair["paper2"]
        structured = (
            CONTRADICTION_STRUCTURED_OUTPUT
            and getattr(llm_client, "provider", "") == "ollama"
        )
        prompt = build_contradiction_prompt(
            paper1, paper2, structured=structured,
            evidence=(pair.get("evidence_1"), pair.get("evidence_2")),
        )
        options = {"json_schema": CONTRADICTION_JSON_SCHEMA} if structured else {}
        print(f"[LLM] Checking pair {index}/{len(candidates)}...")

        try:
            analysis = llm_client.complete(
                prompt,
                max_tokens=400 if structured else 300,
                temperature=0.0,
                **options,
            )
        except Exception as error:
            analysis = f"LLM call failed: {error}"

        parsed = parse_contradiction_response(analysis)
        checked_pairs.append({
            "paper1": paper1,
            "paper2": paper2,
            "shared_concepts": pair.get("shared_concepts"),
            "shared_concept_names": pair.get("shared_concept_names", []),
            "concept_jaccard": pair.get("concept_jaccard"),
            "candidate_score": pair["candidate_score"],
            "candidate_method": pair.get("candidate_method", "concept_overlap"),
            "evidence_1": pair.get("evidence_1"),
            "evidence_2": pair.get("evidence_2"),
            "nli_contradiction": pair.get("nli_contradiction"),
            "llm_analysis": analysis,
            "verdict": parsed["verdict"],
            "confidence": parsed["confidence"],
            "verdict_valid": parsed["valid"],
            "structured_output": structured,
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
