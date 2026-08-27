import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import LLM_MODEL, LLM_MODEL_FALLBACK, GROQ_MAX_RETRIES, TOP_K
from src.pipeline.retrieval import nesy_retrieve, vector_only_retrieve
from src.pipeline.validator import validate_citations
from src.pipeline.prompts import build_review_prompt
from src.utils.groq_client import groq_chat_with_retry


def llm_review(groq_client, driver, query, top_k=TOP_K, *, baseline=False):
    """Generate a literature review via NeSy retrieval + LLM synthesis.

    Parameters
    ----------
    groq_client : groq.Groq instance
    driver      : Neo4j driver
    query       : str — research query
    top_k       : int — papers to retrieve
    baseline    : bool — if True, use vector-only retrieval (no graph expansion)
    """
    mode_label = "BASELINE" if baseline else "REVIEW"
    print(f"\n{'='*60}")
    print(f"[{mode_label} MODE] {query}")
    print(f"{'='*60}")

    if baseline:
        papers = vector_only_retrieve(query, top_k=top_k)
    else:
        papers = nesy_retrieve(driver, query, top_k=top_k)
    print(f"[{'Baseline' if baseline else 'NeSy'}] Retrieved {len(papers)} papers")

    verified = validate_citations(driver, [p["id"] for p in papers if p.get("id")])
    print(f"[Validator] {len(verified)}/{len(papers)} citations verified")

    toon = "title|year|category|abstract\n"
    for p in papers:
        if p.get("id") in verified:
            toon += f"{p['title'][:80]}|{p['year']}|{p['category']}|{p['abstract'][:300]}\n"

    prompt = build_review_prompt(toon, query)

    print("[LLM] Generating answer...")
    try:
        answer = groq_chat_with_retry(
            groq_client, prompt,
            model=LLM_MODEL,
            fallback_model=LLM_MODEL_FALLBACK,
            max_tokens=1024,
            temperature=0.3,
            max_retries=GROQ_MAX_RETRIES,
        )
    except Exception as e:
        answer = f"LLM call failed: {e}"

    print(f"\n{'─'*60}")
    print("ANSWER:")
    print(f"{'─'*60}")
    print(answer)

    return {"query": query, "papers": papers, "answer": answer, "verified": verified}
