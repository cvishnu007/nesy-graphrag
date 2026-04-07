import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import TOP_K
from src.pipeline.retrieval import neural_retrieve


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
        prompt = f"""You are a research hypothesis generator.

CURRENT RESEARCH (papers related to the query):
{query_context}

UNDISCOVERED CONNECTION:
Title: {h['title']}
Year: {h['year']}
Category: {h['category']}
Shared Concepts: {h['shared_concepts']}

This paper shares {h['shared_concepts']} concepts with the query papers
but has NEVER been cited together with them — this is a structural hole
in the knowledge graph.

Generate a research hypothesis in this format:
HYPOTHESIS: [1 clear sentence stating the potential connection]
RATIONALE: [2-3 sentences explaining why combining these could be valuable]
POTENTIAL IMPACT: [1 sentence on what new knowledge this could produce]"""

        print(f"[LLM] Generating hypothesis {i+1}/{len(hypotheses)}...")

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            result = response.choices[0].message.content
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
