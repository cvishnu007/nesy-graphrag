"""
src/pipeline/prompts.py
========================
Centralised prompt templates for the NeSy-GraphRAG LLM pipeline.

Every prompt that is sent to Groq (or any future LLM) should be built
by one of the functions below.  This eliminates the copy-paste drift
that previously existed between the pipeline modules and the Streamlit
UI layer.
"""


def build_review_prompt(toon: str, query: str) -> str:
    """Build the literature-review synthesis prompt.

    Parameters
    ----------
    toon  : str — pipe-separated table of verified papers
    query : str — the user's research query

    Returns
    -------
    str — ready-to-send prompt
    """
    return f"""You are a scientific research assistant specialized in computer science.

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


def build_contradiction_prompt(p1: dict, p2: dict) -> str:
    """Build the contradiction-checking prompt for a paper pair.

    Parameters
    ----------
    p1, p2 : dict — paper dicts with keys: title, year, abstract

    Returns
    -------
    str — ready-to-send prompt
    """
    abs1 = (p1.get("abstract") or "No abstract available.")[:400]
    abs2 = (p2.get("abstract") or "No abstract available.")[:400]

    return f"""You are a scientific fact-checker analyzing research papers.

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


def build_hypothesis_prompt(query_context: str, hypothesis: dict) -> str:
    """Build the hypothesis-generation prompt for a structural hole.

    Parameters
    ----------
    query_context : str — formatted context string of query-related papers
    hypothesis    : dict — candidate paper dict with keys:
                    title, year, category, shared_concepts

    Returns
    -------
    str — ready-to-send prompt
    """
    return f"""You are a research hypothesis generator.

CURRENT RESEARCH (papers related to the query):
{query_context}

UNDISCOVERED CONNECTION:
Title: {hypothesis['title']}
Year: {hypothesis['year']}
Category: {hypothesis['category']}
Shared Concepts: {hypothesis['shared_concepts']}

This paper shares {hypothesis['shared_concepts']} concepts with the query papers
but has NEVER been cited together with them — this is a structural hole
in the knowledge graph.

Generate a research hypothesis in this format:
HYPOTHESIS: [1 clear sentence stating the potential connection]
RATIONALE: [2-3 sentences explaining why combining these could be valuable]
POTENTIAL IMPACT: [1 sentence on what new knowledge this could produce]"""
