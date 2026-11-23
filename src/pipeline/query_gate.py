"""Query gate: refuse inputs that are not computer-science research queries.

Generic or non-scientific inputs ("hello how are you", "best pizza near me")
still return the "nearest" papers from a dense index, and the reasoning modes
would then build reviews, contradictions and hypotheses on unrelated papers.
The gate runs before any mode and stops such inputs with a clear message.

Statistic: ``closeness`` = mean BGE cosine similarity of the query's 5 nearest
papers. Rule:

1. No content words (only stop words, fillers or numbers) -> invalid.
2. closeness >= ``QUERY_GATE_MIN_SIMILARITY`` (0.65) -> valid.
3. closeness >= ``QUERY_GATE_ENTITY_MIN_SIMILARITY`` (0.58) and the query names a
   multi-word scientific entity from the SciERC graph (e.g. "operating systems",
   "intrusion detection") -> valid. Short topic names have lower closeness than
   full questions, so a recognised technical term is accepted as evidence instead.
4. Otherwise -> invalid.

Thresholds come from ``src/evaluation/query_gate_calibration.py``
(``results/query_gate_20261123/summary.json``): 110/110 benchmark, demo and
topic queries pass and 40/40 everyday sentences are rejected; the highest
everyday sentence has closeness 0.605 and no multi-word entity.
"""
from __future__ import annotations

import re

from src.utils.config import (
    QUERY_GATE_ENABLED,
    QUERY_GATE_ENTITY_MIN_SIMILARITY,
    QUERY_GATE_MIN_SIMILARITY,
    QUERY_GATE_TOP_N,
)

# Function words and conversational filler that carry no research content.
STOP_WORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just let me more most my myself no nor
not now of off on once only or other our ours ourselves out over own same she should so some such than that
the their theirs them themselves then there these they this those through to too under until up very was we
were what when where which while who whom why will with would you your yours yourself yourselves
hi hello hey hii thanks thank please ok okay yes yeah yep nope bye good morning evening night today
tomorrow yesterday tell give show want need like know think get got make go going come really much many
something anything everything nothing someone anyone everyone thing things stuff lol hmm umm
""".split())

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-\+\.#]*")
MIN_ENTITY_DF = 5


def _tokens(query: str) -> list[str]:
    return [t.strip(".-") for t in _TOKEN.findall((query or "").lower()) if t.strip(".-")]


def content_words(query: str) -> list[str]:
    """Tokens that could carry research content (3+ characters, not stop words, not numbers)."""
    return [t for t in _tokens(query)
            if len(t) >= 3 and t not in STOP_WORDS and not t.replace(".", "").isdigit()]


def candidate_terms(query: str, max_len: int = 4) -> list[str]:
    """Multi-word phrases (2-4 tokens) that do not start or end with a stop word."""
    toks = _tokens(query)
    out = []
    for n in range(2, max_len + 1):
        for i in range(len(toks) - n + 1):
            gram = toks[i:i + n]
            if gram[0] in STOP_WORDS or gram[-1] in STOP_WORDS:
                continue
            out.append(" ".join(gram))
    return sorted(set(out))


def gate_decision(query: str, similarities, entity_terms=(), *,
                  min_similarity: float = QUERY_GATE_MIN_SIMILARITY,
                  entity_min_similarity: float = QUERY_GATE_ENTITY_MIN_SIMILARITY,
                  top_n: int = QUERY_GATE_TOP_N) -> dict:
    """Decide validity from the query, its nearest papers' similarities (best first) and
    the multi-word scientific entities it names. Returns valid, reason, message and the evidence."""
    words = content_words(query)
    top = [float(s) for s in list(similarities)[:top_n]]
    closeness = round(sum(top) / len(top), 4) if top else 0.0
    entities = sorted(entity_terms)
    if not words:
        reason = "no_content_words"
    elif closeness >= min_similarity:
        reason = "ok"
    elif closeness >= entity_min_similarity and entities:
        reason = "ok_scientific_term"
    else:
        reason = "off_topic"
    valid = reason.startswith("ok")
    if reason == "no_content_words":
        message = ("Not a valid query: it has no subject words. Ask about a computer-science topic, "
                   "method or problem, for example \"graph neural networks for fraud detection\".")
    elif not valid:
        message = ("Not a valid query: it does not match any computer-science research in the corpus "
                   f"(closeness {closeness:.2f}; needed {min_similarity:.2f}, or {entity_min_similarity:.2f} "
                   "with a recognised technical term). Ask about a computer-science topic, method or problem.")
    else:
        message = ""
    return {"valid": valid, "reason": reason, "message": message, "closeness": closeness,
            "content_words": words, "scientific_terms": entities}


def scientific_terms(driver, query: str) -> list[str]:
    """Multi-word phrases of the query that are SciERC entities in the graph (document frequency >= 5)."""
    terms = candidate_terms(query)
    if not terms or driver is None:
        return []
    with driver.session() as session:
        rows = session.run("UNWIND $terms AS t MATCH (e:SciEntity {name: t}) WHERE coalesce(e.df, 0) >= $min_df "
                           "RETURN e.name AS name", terms=terms, min_df=MIN_ENTITY_DF)
        return sorted(r["name"] for r in rows)


def check_query(query: str, driver=None) -> dict:
    """Run the gate against the live stores. Always valid when the gate is disabled."""
    if not QUERY_GATE_ENABLED:
        return {"valid": True, "reason": "gate_disabled", "message": "", "closeness": None,
                "content_words": content_words(query), "scientific_terms": []}
    if not content_words(query):
        return gate_decision(query, [])
    from src.storage.chroma_store import query as chroma_query

    sims = [p["score"] for p in chroma_query(query, top_k=QUERY_GATE_TOP_N)]
    terms = []
    if sum(sims[:QUERY_GATE_TOP_N]) / max(1, len(sims[:QUERY_GATE_TOP_N])) < QUERY_GATE_MIN_SIMILARITY:
        terms = scientific_terms(driver, query)  # only needed for borderline queries
    return gate_decision(query, sims, terms)
