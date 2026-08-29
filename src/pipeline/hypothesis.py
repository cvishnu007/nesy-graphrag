import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.pipeline.prompts import build_hypothesis_prompt
from src.pipeline.retrieval import neural_retrieve
from src.utils.config import (
    GROQ_MAX_RETRIES,
    HYPOTHESIS_CANDIDATE_POOL,
    HYPOTHESIS_MIN_QUERY_SUPPORT,
    HYPOTHESIS_MIN_SHARED_CONCEPTS,
    LLM_MODEL,
    LLM_MODEL_FALLBACK,
)
from src.utils.groq_client import groq_chat_with_retry


def score_hypothesis_candidate(
    shared_concepts,
    candidate_concepts,
    supporting_papers,
    query_papers,
):
    """Return concept overlap, query support, and combined evidence score."""
    overlap = shared_concepts / candidate_concepts if candidate_concepts > 0 else 0.0
    support_ratio = supporting_papers / query_papers if query_papers > 0 else 0.0
    score = 0.6 * overlap + 0.4 * support_ratio
    return overlap, support_ratio, score


def _field(text, name):
    cleaned = (text or "").replace("**", "")
    match = re.search(
        rf"^\s*{re.escape(name)}\s*:\s*(.*?)"
        rf"(?=^\s*[A-Z][A-Z ]+\s*:|\Z)",
        cleaned,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def parse_hypothesis_response(text):
    """Parse feasibility and evidence fields from generated hypothesis text."""
    cleaned = (text or "").replace("**", "")
    feasibility_match = re.search(
        r"^\s*FEASIBILITY\s*:\s*(HIGH|MEDIUM|LOW)\s*$",
        cleaned,
        re.IGNORECASE | re.MULTILINE,
    )
    feasibility = feasibility_match.group(1).upper() if feasibility_match else "UNKNOWN"
    hypothesis = _field(text, "HYPOTHESIS")
    supporting_evidence = _field(text, "SUPPORTING EVIDENCE")
    missing_evidence = _field(text, "MISSING EVIDENCE")
    return {
        "hypothesis": hypothesis,
        "feasibility": feasibility,
        "supporting_evidence": supporting_evidence,
        "missing_evidence": missing_evidence,
        "valid": bool(
            hypothesis
            and feasibility != "UNKNOWN"
            and supporting_evidence
            and missing_evidence
        ),
    }


def partition_validated_hypotheses(hypotheses):
    """Separate accepted hypotheses while preserving rejected items for audit."""
    accepted = [item for item in hypotheses if item.get("accepted")]
    rejected = [item for item in hypotheses if not item.get("accepted")]
    return accepted, rejected


def generate_hypotheses(driver, query, top_k=5):
    """Rank uncited structural holes by normalized graph evidence."""
    neural_ids = [paper["id"] for paper in neural_retrieve(query, top_k=5)]
    if not neural_ids:
        return []

    cypher = """
        UNWIND $ids AS pid
        MATCH (query_paper:Paper {id: pid})-[:RELATED_TO]->(shared:Concept)
              <-[:RELATED_TO]-(candidate:Paper)
        WHERE NOT candidate.id IN $ids
        WITH candidate,
             collect(DISTINCT shared.name) AS shared_names,
             collect(DISTINCT query_paper.id) AS supporting_ids
        WHERE size(shared_names) >= $min_shared
          AND size(supporting_ids) >= $min_support
          AND NOT EXISTS {
              MATCH (query_paper:Paper)-[:CITES]-(candidate)
              WHERE query_paper.id IN $ids
          }
        OPTIONAL MATCH (candidate)-[:RELATED_TO]->(candidate_concept:Concept)
        WITH candidate, shared_names, supporting_ids,
             count(DISTINCT candidate_concept) AS candidate_concepts
        RETURN candidate.id AS id,
               candidate.title AS title,
               candidate.year AS year,
               candidate.category AS category,
               shared_names,
               supporting_ids,
               candidate_concepts
        ORDER BY size(supporting_ids) DESC, size(shared_names) DESC
        LIMIT $candidate_pool
    """
    with driver.session() as session:
        records = list(session.run(
            cypher,
            ids=neural_ids,
            min_shared=HYPOTHESIS_MIN_SHARED_CONCEPTS,
            min_support=HYPOTHESIS_MIN_QUERY_SUPPORT,
            candidate_pool=max(top_k, HYPOTHESIS_CANDIDATE_POOL),
        ))

    candidates = []
    for record in records:
        shared_count = len(record["shared_names"])
        support_count = len(record["supporting_ids"])
        overlap, support_ratio, score = score_hypothesis_candidate(
            shared_count,
            record["candidate_concepts"],
            support_count,
            len(neural_ids),
        )
        candidates.append({
            "id": record["id"],
            "title": record["title"],
            "year": record["year"],
            "category": record["category"],
            "shared_concepts": shared_count,
            "shared_concept_names": record["shared_names"],
            "supporting_paper_ids": record["supporting_ids"],
            "supporting_papers": support_count,
            "concept_overlap": round(overlap, 4),
            "query_support_ratio": round(support_ratio, 4),
            "evidence_score": round(score, 4),
            "hypothesis": (
                f"'{record['title']}' is an uncited structural hole supported by "
                f"{support_count} query papers and {shared_count} shared concepts"
            ),
        })

    return sorted(
        candidates,
        key=lambda item: (-item["evidence_score"], -item["supporting_papers"]),
    )[:top_k]


def llm_hypothesis(groq_client, driver, query, top_k=5):
    print(f"\n{'=' * 60}")
    print(f"[HYPOTHESIS MODE] {query}")
    print(f"{'=' * 60}")

    hypotheses = generate_hypotheses(driver, query, top_k=top_k)
    query_papers = neural_retrieve(query, top_k=3)
    print(f"[Graph] Found {len(hypotheses)} evidence-ranked structural holes")
    if not hypotheses:
        print("No hypothesis candidates found.")
        return {"query": query, "hypotheses": []}

    query_context = "\n".join(
        f"- {paper['title'][:80]} ({paper['year']}): {(paper.get('abstract') or '')[:200]}"
        for paper in query_papers
    )
    enriched = []
    for index, hypothesis in enumerate(hypotheses, 1):
        print(f"[LLM] Generating hypothesis {index}/{len(hypotheses)}...")
        try:
            response = groq_chat_with_retry(
                groq_client,
                build_hypothesis_prompt(query_context, hypothesis),
                model=LLM_MODEL,
                fallback_model=LLM_MODEL_FALLBACK,
                max_tokens=700,
                temperature=0.0,
                max_retries=GROQ_MAX_RETRIES,
            )
        except Exception as error:
            response = f"LLM call failed: {error}"

        parsed = parse_hypothesis_response(response)
        enriched.append({
            "paper": hypothesis,
            "llm_hypothesis": response,
            "feasibility": parsed["feasibility"],
            "supporting_evidence": parsed["supporting_evidence"],
            "missing_evidence": parsed["missing_evidence"],
            "validation_valid": parsed["valid"],
            "accepted": parsed["valid"] and parsed["feasibility"] != "LOW",
        })
        print(
            f"  Hypothesis {index}: feasibility={parsed['feasibility']}, "
            f"valid={parsed['valid']}"
        )

    accepted, rejected = partition_validated_hypotheses(enriched)
    return {
        "query": query,
        "hypotheses": accepted,
        "rejected_hypotheses": rejected,
        "total_candidates": len(enriched),
    }
