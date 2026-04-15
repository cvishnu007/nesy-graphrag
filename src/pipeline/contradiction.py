import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import LLM_MODEL, TOP_K
from src.pipeline.retrieval import nesy_retrieve


def detect_contradictions(driver, query, top_k=5):
    """Find paper pairs from different years sharing >= 2 concepts."""
    papers    = nesy_retrieve(driver, query, top_k=10)
    paper_ids = [p["id"] for p in papers if p.get("id")]

    if len(papers) < 2:
        return []

    with driver.session() as session:
        result = session.run("""
            UNWIND $ids AS id1
            UNWIND $ids AS id2
            WITH id1, id2
            WHERE id1 < id2
            MATCH (p1:Paper {id: id1})-[:RELATED_TO]->(c:Concept)<-[:RELATED_TO]-(p2:Paper {id: id2})
            WITH p1, p2, count(c) AS shared
            WHERE shared >= 2 AND p1.year <> p2.year
            RETURN p1.id AS id1, p1.title AS title1, p1.abstract AS abstract1, p1.year AS year1,
                   p2.id AS id2, p2.title AS title2, p2.abstract AS abstract2, p2.year AS year2,
                   shared
            ORDER BY shared DESC
            LIMIT $top_k
        """, ids=paper_ids, top_k=top_k)

        contradictions = []
        for r in result:
            contradictions.append({
                "paper1": {"id": r["id1"], "title": r["title1"],
                           "abstract": r["abstract1"], "year": r["year1"]},
                "paper2": {"id": r["id2"], "title": r["title2"],
                           "abstract": r["abstract2"], "year": r["year2"]},
                "shared_concepts": r["shared"],
                "flag": f"Papers from {r['year1']} and {r['year2']} share {r['shared']} concepts — potential contradiction"
            })
    return contradictions


def llm_contradict(groq_client, driver, query, top_k=5):
    print(f"\n{'='*60}")
    print(f"[CONTRADICTION MODE] {query}")
    print(f"{'='*60}")

    contradictions = detect_contradictions(driver, query, top_k=top_k)
    print(f"[Graph] Found {len(contradictions)} candidate pairs")

    if not contradictions:
        print("No contradiction candidates found.")
        return {"query": query, "contradictions": []}

    verified_contradictions = []

    for i, pair in enumerate(contradictions):
        p1   = pair["paper1"]
        p2   = pair["paper2"]
        abs1 = (p1.get("abstract") or "No abstract available.")[:400]
        abs2 = (p2.get("abstract") or "No abstract available.")[:400]

        prompt = f"""You are a scientific fact-checker analyzing research papers.

Compare these two papers and determine if they CONTRADICT each other:

PAPER 1 ({p1['year']}): {p1['title']}
Abstract: {abs1}

PAPER 2 ({p2['year']}): {p2['title']}
Abstract: {abs2}

Answer in this exact format:
VERDICT: [CONTRADICTION / AGREEMENT / DIFFERENT SCOPE]
REASON: [1-2 sentences explaining why]
CLAIM 1: [What Paper 1 claims]
CLAIM 2: [What Paper 2 claims]"""

        print(f"[LLM] Checking pair {i+1}/{len(contradictions)}...")

        try:
            response = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            result = response.choices[0].message.content
        except Exception as e:
            result = f"LLM call failed: {e}"

        verified_contradictions.append({
            "paper1"      : p1,
            "paper2"      : p2,
            "llm_analysis": result
        })

        print(f"\n  Pair {i+1}:")
        print(f"  P1: {p1['title'][:60]}... ({p1['year']})")
        print(f"  P2: {p2['title'][:60]}... ({p2['year']})")
        print(f"  {result[:200]}...")

    return {"query": query, "contradictions": verified_contradictions}
