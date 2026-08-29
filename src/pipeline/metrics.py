"""
src/pipeline/metrics.py
=======================
NeSy-GraphRAG Evaluation Metrics — Phase 3

Implements five prototype diagnostics that can be computed without human annotation
and with the current pipeline state (Groq + Neo4j may be unavailable;
ChromaDB must be up).

Metrics
-------
1. TS  — Trustworthiness Score
2. NBR — NeSy Boost Ratio
3. ATD — Answer Temporal Diversity
4. RDI — Reasoning Depth Index
5. HNS — Hypothesis Novelty Score

Usage
-----
    from src.pipeline.metrics import compute_all_metrics

    result = graphrag_query(query, mode="review")   # or "contradict"
    scores = compute_all_metrics(result)
    print(scores)
"""

import os
import sys
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipeline.verdicts import is_confident_contradiction
from src.utils.config import (
    CONTRADICTION_MIN_CONFIDENCE,
    EVALUATION_END_YEAR,
    EVALUATION_START_YEAR,
)


# ─────────────────────────────────────────────────────────────
# 1. TRUSTWORTHINESS SCORE (TS)
# ─────────────────────────────────────────────────────────────

def compute_provenance_ts(provenance: dict) -> dict:
    """Compute prototype trustworthiness from validated claim provenance."""
    stats = provenance.get("stats", {})
    total_claims = int(stats.get("total_claims", 0))
    total_citations = int(stats.get("total_citations", 0))
    valid_citations = int(stats.get("valid_citations", 0))
    grounded_claims = int(stats.get("grounded_claims", 0))

    citation_integrity = (
        valid_citations / total_citations if total_citations else 0.0
    )
    hallucination_rate = (
        (total_citations - valid_citations) / total_citations
        if total_citations
        else 0.0
    )
    claim_coverage = grounded_claims / total_claims if total_claims else 0.0
    ts = 0.5 * citation_integrity + 0.5 * claim_coverage

    return {
        "ts": round(ts, 4),
        "citation_integrity": round(citation_integrity, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "claim_coverage": round(claim_coverage, 4),
        "total_claims": total_claims,
        "grounded_claims": grounded_claims,
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "hallucinated_count": total_citations - valid_citations,
    }

def compute_ts(verified: dict, papers: list, answer: str) -> dict:
    """
    TS = 0.5 * Citation_Integrity + 0.5 * (1 - Hallucination_Rate)

    Citation_Integrity  = |Verified IDs in Neo4j| / |Total IDs cited by LLM|
    Hallucination_Rate  = |Titles in answer NOT in verified set| / |Titles in answer|

    Parameters
    ----------
    verified : dict  {paper_id: paper_title}  — output of validate_citations()
    papers   : list  — full list of retrieved papers (from nesy_retrieve)
    answer   : str   — raw LLM answer text

    Returns
    -------
    dict with ts, citation_integrity, hallucination_rate, detail
    """
    total_retrieved = len(papers)
    total_verified  = len(verified)

    if total_retrieved == 0:
        return {
            "ts": 0.0,
            "citation_integrity": 0.0,
            "hallucination_rate": 0.0,
            "total_retrieved": 0,
            "total_verified": 0,
            "hallucinated_count": 0,
        }

    # Citation Integrity: what fraction of retrieved papers actually exist in Neo4j
    citation_integrity = total_verified / total_retrieved if total_retrieved > 0 else 0.0

    # Hallucination Rate: scan the LLM answer for paper titles, check if each is verified
    verified_titles_lower = {t.lower() for t in verified.values()}
    answer_lower          = answer.lower() if answer else ""

    cited_in_answer   = [t for t in verified_titles_lower if t[:40] in answer_lower]
    # Also catch any title fragments that appear in the answer but are NOT verified
    all_titles_lower  = {p["title"].lower() for p in papers if p.get("title")}
    unverified_titles = all_titles_lower - verified_titles_lower
    hallucinated      = [t for t in unverified_titles if t[:40] in answer_lower]

    total_title_mentions = len(cited_in_answer) + len(hallucinated)
    hallucination_rate   = (
        len(hallucinated) / total_title_mentions
        if total_title_mentions > 0
        else 0.0
    )

    ts = 0.5 * citation_integrity + 0.5 * (1.0 - hallucination_rate)

    return {
        "ts"                 : round(ts, 4),
        "citation_integrity" : round(citation_integrity, 4),
        "hallucination_rate" : round(hallucination_rate, 4),
        "total_retrieved"    : total_retrieved,
        "total_verified"     : total_verified,
        "hallucinated_count" : len(hallucinated),
    }


# ─────────────────────────────────────────────────────────────
# 2. NESY BOOST RATIO (NBR)
# ─────────────────────────────────────────────────────────────

def compute_nbr(papers: list) -> dict:
    """
    NBR = |Papers from graph expansion| / |Total retrieved|

    Papers with source == "symbolic" or "both" came from Neo4j graph expansion.
    Papers with source == "neural" came from ChromaDB only.

    Target: NBR > 0.3 — proves Neo4j adds value beyond ChromaDB alone.

    Parameters
    ----------
    papers : list — output of nesy_retrieve(); each dict has a 'source' key:
                    "neural" | "symbolic" | "both"

    Returns
    -------
    dict with nbr, graph_count, neural_only_count, total
    """
    total       = len(papers)
    graph_count = sum(1 for p in papers if p.get("source") in {"symbolic", "both"})

    nbr = graph_count / total if total > 0 else 0.0

    return {
        "nbr"              : round(nbr, 4),
        "graph_count"      : graph_count,
        "neural_only_count": total - graph_count,
        "total"            : total,
        "graph_contributes": graph_count > 0,
    }


# ─────────────────────────────────────────────────────────────
# 3. ANSWER TEMPORAL DIVERSITY (ATD)
# ─────────────────────────────────────────────────────────────

def compute_atd(papers: list, year_range: tuple | None = None) -> dict:
    """
    ATD = |Distinct years in cited papers| / span_size

    The default range comes from EVALUATION_START_YEAR and
    EVALUATION_END_YEAR.

    Parameters
    ----------
    papers     : list — retrieved papers, each with a 'year' key
    year_range : tuple — (start_year, end_year) inclusive

    Returns
    -------
    dict with atd, distinct_years, year_distribution
    """
    start, end = year_range or (EVALUATION_START_YEAR, EVALUATION_END_YEAR)
    span_size   = end - start + 1

    years_in_results = [
        int(p["year"])
        for p in papers
        if p.get("year") and str(p["year"]).isdigit()
        and start <= int(p["year"]) <= end
    ]

    distinct_years   = set(years_in_results)
    year_counts      = {y: years_in_results.count(y) for y in sorted(distinct_years)}

    atd = len(distinct_years) / span_size if span_size > 0 else 0.0

    return {
        "atd"             : round(atd, 4),
        "distinct_years"  : sorted(distinct_years),
        "year_distribution": year_counts,
        "span_size"       : span_size,
        "missing_years"   : [y for y in range(start, end + 1) if y not in distinct_years],
    }


# ─────────────────────────────────────────────────────────────
# 4. REASONING DEPTH INDEX (RDI)
# ─────────────────────────────────────────────────────────────

def compute_rdi(
    papers: list,
    contradictions: list,
    total_possible_contradictions: int | None = None,
) -> dict:
    """
    RDI = (Sources_Used + Contradictions_Detected)
          / (Total_Sources + Total_Contradictions)

    Sources_Used            = papers cited from more than one retrieval source
    Contradictions_Detected = pairs with LLM verdict = "CONTRADICTION"
    Total_Sources           = total papers retrieved
    Total_Contradictions    = total contradiction candidates checked
                              (defaults to len(contradictions) if not provided)

    Parameters
    ----------
    papers                         : list — retrieved papers
    contradictions                 : list — output of detect_contradictions() or
                                     llm_contradict()["contradictions"]
    total_possible_contradictions  : int  — how many pairs were checked in total
                                     (pass len(contradictions) if unknown)

    Returns
    -------
    dict with rdi, cross_doc_support, contradiction_resolution_rate
    """
    total_sources = len(papers)

    # Cross-document support: papers that were reached via BOTH neural and symbolic
    cross_doc = sum(1 for p in papers if p.get("source") == "both")
    cross_doc_support = cross_doc / total_sources if total_sources > 0 else 0.0

    # Contradiction resolution rate
    # Handles both raw detect_contradictions() output and llm_contradict() output
    resolved = 0
    for item in contradictions:
        if is_confident_contradiction(item, CONTRADICTION_MIN_CONFIDENCE):
            resolved += 1

    total_checked = (
        total_possible_contradictions
        if total_possible_contradictions is not None
        else len(contradictions)
    )
    contradiction_resolution_rate = (
        resolved / total_checked if total_checked > 0 else 0.0
    )

    # Final RDI
    numerator   = cross_doc + resolved
    denominator = total_sources + total_checked
    rdi = numerator / denominator if denominator > 0 else 0.0

    return {
        "rdi"                          : round(rdi, 4),
        "cross_doc_support"            : round(cross_doc_support, 4),
        "contradiction_resolution_rate": round(contradiction_resolution_rate, 4),
        "cross_doc_papers"             : cross_doc,
        "contradictions_resolved"      : resolved,
        "total_sources"                : total_sources,
        "total_checked"                : total_checked,
    }


# ─────────────────────────────────────────────────────────────
# 5. HYPOTHESIS NOVELTY SCORE (HNS)
# ─────────────────────────────────────────────────────────────

def compute_hns(
    driver,
    hypotheses: list,
    query_paper_ids: list,
) -> dict:
    """
    HNS = mean(shortestPath_length / maximum_path_length) across hypotheses.

    For each hypothesis paper, find the shortest path through Concept
    nodes between any of its concepts and any concept belonging to the
    query papers. Longer shortest paths produce higher structural novelty.

    If Neo4j is unavailable or no paths exist, returns hns = 0.0.

    Parameters
    ----------
    driver           : Neo4j driver (may be None)
    hypotheses       : list — output of llm_hypothesis()["hypotheses"]
                       Each item has a 'paper' dict with an 'id' key.
    query_paper_ids  : list — IDs of the user's query papers

    Returns
    -------
    dict with hns, individual_scores, total_hypotheses
    """
    if not driver or not hypotheses or not query_paper_ids:
        return {
            "hns": 0.0,
            "individual_scores": [],
            "total_hypotheses": len(hypotheses) if hypotheses else 0,
        }

    # Extract hypothesis paper IDs
    hyp_ids = []
    for h in hypotheses:
        paper = h.get("paper", h)  # handle both enriched and raw format
        pid = paper.get("id")
        if pid:
            hyp_ids.append(pid)

    if not hyp_ids:
        return {
            "hns": 0.0,
            "individual_scores": [],
            "total_hypotheses": len(hypotheses),
        }

    individual_scores = []

    try:
        with driver.session() as session:
            for hid in hyp_ids:
                result = session.run("""
                    MATCH (h:Paper {id: $hid})-[:RELATED_TO]->(hc:Concept)
                    MATCH (q:Paper)-[:RELATED_TO]->(qc:Concept)
                    WHERE q.id IN $qids AND hc <> qc
                    MATCH path = shortestPath((hc)-[*..6]-(qc))
                    WITH length(path) AS pathLen
                    ORDER BY pathLen ASC
                    LIMIT 1
                    RETURN pathLen
                """, hid=hid, qids=query_paper_ids)

                record = result.single()
                if record and record["pathLen"] > 0:
                    individual_scores.append(min(float(record["pathLen"]), 6.0) / 6.0)
                else:
                    # Missing paths provide no measurable novelty evidence.
                    individual_scores.append(0.0)

    except Exception as e:
        print(f"[HNS] Neo4j query failed: {e}")
        return {
            "hns": 0.0,
            "individual_scores": [],
            "total_hypotheses": len(hypotheses),
        }

    hns = (
        sum(individual_scores) / len(individual_scores)
        if individual_scores
        else 0.0
    )

    return {
        "hns": round(hns, 4),
        "individual_scores": [round(s, 4) for s in individual_scores],
        "total_hypotheses": len(hypotheses),
    }


# ─────────────────────────────────────────────────────────────
# COMBINED RUNNER
# ─────────────────────────────────────────────────────────────

def compute_all_metrics(
    result: dict,
    contradiction_result: dict | None = None,
    hypothesis_result: dict | None = None,
    driver=None,
    year_range: tuple | None = None,
) -> dict[str, Any]:
    """
    Compute all five prototype diagnostics from a pipeline result dict.

    Parameters
    ----------
    result               : dict — output of graphrag_query(mode="review")
                           Must have keys: papers, answer, verified
    contradiction_result : dict — output of graphrag_query(mode="contradict")
    hypothesis_result    : dict — output of graphrag_query(mode="hypothesis")
                           Optional. If provided, HNS is computed.
    driver               : Neo4j driver — required for HNS computation
                           Optional. If None, RDI is computed with 0 contradictions.
    year_range           : tuple — dataset year span for ATD normalisation

    Returns
    -------
    dict with keys: ts, nbr, atd, rdi, hns (each is itself a dict)
    """
    papers      = result.get("papers", [])
    answer      = result.get("answer", "")
    verified    = result.get("verified", {})

    contradictions = []
    if contradiction_result:
        contradictions = contradiction_result.get("contradictions", [])

    provenance = result.get("provenance")
    ts = (
        compute_provenance_ts(provenance)
        if provenance is not None
        else compute_ts(verified, papers, answer)
    )
    nbr = compute_nbr(papers)
    atd = compute_atd(papers, year_range=year_range)
    rdi = compute_rdi(papers, contradictions)

    # HNS — compute only if hypothesis results and a driver are available
    hypotheses      = hypothesis_result.get("hypotheses", []) if hypothesis_result else []
    query_paper_ids = [p["id"] for p in papers if p.get("id")]
    hns = compute_hns(driver, hypotheses, query_paper_ids)

    scores = {
        "ts" : ts,
        "nbr": nbr,
        "atd": atd,
        "rdi": rdi,
        "hns": hns,
    }

    _print_summary(scores)
    return scores


def _print_summary(scores: dict) -> None:
    """Pretty-print a metrics summary to stdout."""
    ts  = scores["ts"]
    nbr = scores["nbr"]
    atd = scores["atd"]
    rdi = scores["rdi"]
    hns = scores.get("hns", {})

    print("\n" + "=" * 55)
    print("  NeSy-GraphRAG EVALUATION METRICS")
    print("=" * 55)
    print(f"  TS  (Trustworthiness)   : {ts['ts']:.4f}"
          f"  [CI={ts['citation_integrity']:.2f}, HR={ts['hallucination_rate']:.2f}]")
    print(f"  NBR (NeSy Boost Ratio)  : {nbr['nbr']:.4f}"
          f"  [{nbr['graph_count']}/{nbr['total']} include graph retrieval]")
    print(f"  ATD (Temporal Diversity): {atd['atd']:.4f}"
          f"  [years: {atd['distinct_years']}]")
    print(f"  RDI (Reasoning Depth)   : {rdi['rdi']:.4f}"
          f"  [cross-doc={rdi['cross_doc_papers']}, contradictions={rdi['contradictions_resolved']}]")
    if hns and hns.get("hns") is not None:
        print(f"  HNS (Hypothesis Novelty): {hns['hns']:.4f}"
              f"  [{hns.get('total_hypotheses', 0)} hypotheses scored]")
    print("=" * 55 + "\n")
