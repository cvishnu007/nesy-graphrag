import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import TOP_K
from src.pipeline.retrieval import nesy_retrieve
from src.pipeline.validator import validate_citations


def llm_review(groq_client, driver, query, top_k=TOP_K):
    print(f"\n{'='*60}")
    print(f"[REVIEW MODE] {query}")
    print(f"{'='*60}")

    papers = nesy_retrieve(driver, query, top_k=top_k)
    print(f"[NeSy] Retrieved {len(papers)} papers")

    verified = validate_citations(driver, [p["id"] for p in papers if p.get("id")])
    print(f"[Validator] {len(verified)}/{len(papers)} citations verified")

    toon = "title|year|category|abstract\n"
    for p in papers:
        if p.get("id") in verified:
            toon += f"{p['title'][:80]}|{p['year']}|{p['category']}|{p['abstract'][:300]}\n"

    prompt = f"""You are a scientific research assistant specialized in computer science.

Below are research papers in TOON format (pipe-separated):
title|year|category|abstract

PAPERS:
{toon}

QUERY: {query}

Your task:
1. Write a clear 2-3 paragraph synthesis answering the query
2. Cite papers by their title in [brackets]
3. Highlight key findings and trends across years
4. End with a 1-line summary of the state of the field

Be precise and academic in tone."""

    print("[LLM] Generating answer...")
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3
    )
    answer = response.choices[0].message.content

    print(f"\n{'─'*60}")
    print("ANSWER:")
    print(f"{'─'*60}")
    print(answer)

    return {"query": query, "papers": papers, "answer": answer, "verified": verified}
