import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import LLM_MODEL, LLM_MODEL_FALLBACK, GROQ_MAX_RETRIES, TOP_K
from src.pipeline.retrieval import neural_retrieve
from src.pipeline.prompts import build_hypothesis_prompt
from src.utils.groq_client import groq_chat_with_retry


def generate_hypotheses(driver, query, top_k=5):
    """Find structural holes — papers sharing concepts but never cited together."""
    neural_ids = [p["id"] for p in neural_retrieve(query, top_k=5)]

    with driver.session() as session:
        result = session.run("""
            UNWIND $ids AS pid
            MATCH (p:Paper {id: pid})-[:RELATED_TO]->(c:Concept)
                  <-[:RELATED_TO]-(candidate:Paper)
            WHERE NOT candidate.id IN $ids
            WITH candidate, count(c) AS shared_concepts
            WHERE shared_concepts >= 2
            AND NOT EXISTS {
                MATCH (p2:Paper)-[:CITES]-(candidate)
                WHERE p2.id IN $ids
            }
            RETURN candidate.id       AS id,
                   candidate.title    AS title,
                   candidate.year     AS year,
                   candidate.category AS category,
                   shared_concepts
            ORDER BY shared_concepts DESC
            LIMIT $top_k
        """, ids=neural_ids, top_k=top_k)

        hypotheses = []
        for r in result:
            hypotheses.append({
                "id"             : r["id"],
                "title"          : r["title"],
                "year"           : r["year"],
                "category"       : r["category"],
                "shared_concepts": r["shared_concepts"],
                "hypothesis"     : f"'{r['title']}' shares {r['shared_concepts']} concepts with your query papers but has never been cited together — potential research connection"
            })
    return hypotheses


def llm_hypothesis(groq_client, driver, query, top_k=5):
    print(f"\n{'='*60}")
    print(f"[HYPOTHESIS MODE] {query}")
    print(f"{'='*60}")

    hypotheses   = generate_hypotheses(driver, query, top_k=top_k)
    query_papers = neural_retrieve(query, top_k=3)
    print(f"[Graph] Found {len(hypotheses)} structural holes")

    if not hypotheses:
        print("No hypothesis candidates found.")
        return {"query": query, "hypotheses": []}

    query_context = "\n".join([
        f"- {p['title'][:80]} ({p['year']}): {(p.get('abstract') or '')[:200]}"
        for p in query_papers
    ])

    enriched_hypotheses = []

    for i, h in enumerate(hypotheses):
        prompt = build_hypothesis_prompt(query_context, h)

        print(f"[LLM] Generating hypothesis {i+1}/{len(hypotheses)}...")

        try:
            result = groq_chat_with_retry(
                groq_client, prompt,
                model=LLM_MODEL,
                fallback_model=LLM_MODEL_FALLBACK,
                max_tokens=300,
                temperature=0.3,
                max_retries=GROQ_MAX_RETRIES,
            )
        except Exception as e:
            result = f"LLM call failed: {e}"

        enriched_hypotheses.append({
            "paper"         : h,
            "llm_hypothesis": result
        })

        print(f"\n  Hypothesis {i+1}:")
        print(f"  Paper: {h['title'][:60]}...")
        print(f"  {result[:200]}...")

    return {"query": query, "hypotheses": enriched_hypotheses}
