import json
import os
import sys

import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline.orchestrator import get_neo4j, get_llm
from src.pipeline.review import llm_review
from src.pipeline.contradiction import llm_contradict
from src.pipeline.verdicts import contradiction_verdict
from src.pipeline.hypothesis import llm_hypothesis
from src.storage.chroma_store import get_collection
from src.utils.config import (
    CHROMA_COLLECTION, CONCEPT_SOURCE, CONTRADICTION_CANDIDATE_MODE, DATA_SOURCE, EMBEDDING_MODEL,
)
from src.utils.llm_client import llm_runtime_metadata

# ════════════════════════════════════════════════════
# INIT (cached so it only runs once)
# ════════════════════════════════════════════════════
@st.cache_resource
def load_resources():
    driver     = get_neo4j()
    llm_client = get_llm()
    collection = get_collection()
    return driver, llm_client, collection

driver, llm_client, collection = load_resources()
llm_metadata = llm_runtime_metadata()


def graph_stats(driver):
    try:
        with driver.session() as session:
            papers = session.run("MATCH (p:Paper) RETURN count(p) AS c").single()["c"]
            edges = session.run("MATCH (:Paper)-[r:CITES]->(:Paper) RETURN count(r) AS c").single()["c"]
        return papers, edges
    except Exception:
        return "N/A", "N/A"

# ════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════
st.set_page_config(page_title="NeSy-GraphRAG", page_icon="🔬", layout="wide")

st.title("🔬 NeSy-GraphRAG — Research Assistant")
st.caption("Neural + Symbolic Graph Retrieval Augmented Generation")

# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio(
        "Select Mode",
        ["📚 Literature Review", "⚡ Contradiction Detection", "💡 Hypothesis Generation"],
        index=0
    )
    if "review" in mode.lower():
        top_k = st.slider("Papers to retrieve", min_value=3, max_value=15, value=10)
    else:
        top_k = st.slider("Candidates to evaluate", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.markdown("**Pipeline:**")
    paper_nodes, cites_edges = graph_stats(driver)
    st.markdown(f"- Source: `{DATA_SOURCE}`")
    st.markdown(f"- Chroma collection: `{CHROMA_COLLECTION}` ({collection.count()} vectors)")
    st.markdown(f"- Neo4j: {paper_nodes} papers, {cites_edges} CITES edges")
    st.markdown(f"- LLM: `{llm_metadata['provider']} / {llm_metadata['model']}`")
    st.markdown(f"- Embedding: `{EMBEDDING_MODEL}`")
    st.markdown(f"- Concepts: `{CONCEPT_SOURCE}` · contradiction candidates: `{CONTRADICTION_CANDIDATE_MODE}`")

# ── Main ─────────────────────────────────────────────
query = st.text_input(
    "🔍 Enter your research query",
    placeholder="e.g. graph neural networks for node classification"
)

run = st.button("🚀 Run Query", type="primary")

if run and query:

    # ── REVIEW MODE ──────────────────────────────────
    if "review" in mode.lower():
        with st.spinner("Running NeSy retrieval + LLM synthesis..."):
            result = llm_review(llm_client, driver, query, top_k=top_k)

        papers   = result["papers"]
        answer   = result["answer"]
        verified = result["verified"]
        claims = result.get("claims", [])
        unsupported_claims = result.get("unsupported_claims", [])
        provenance = result.get("provenance", {})
        provenance_stats = provenance.get("stats", {})
        parse_errors = provenance.get("parse_errors", [])

        st.success(
            f"Retrieved {len(papers)} papers — {len(verified)}/{len(papers)} papers verified — "
            f"{len(claims)}/{provenance_stats.get('total_claims', 0)} claims have valid passage references"
        )

        st.markdown("### 📝 Literature Review")
        st.markdown(answer)

        st.markdown("### Claim Evidence")
        if not claims:
            st.warning("No claim with a completely valid passage citation set was produced.")
        for index, claim in enumerate(claims, start=1):
            with st.expander(f"Claim {index}: {claim['text'][:90]}"):
                for evidence in claim.get("evidence", []):
                    st.markdown(
                        f"**{evidence['id']} — {evidence['paper_title']}**"
                    )
                    st.write(evidence["text"])

        if unsupported_claims:
            st.warning(
                f"Blocked {len(unsupported_claims)} claim(s) with missing or invalid evidence."
            )
            with st.expander("Unsupported claim audit"):
                for claim in unsupported_claims:
                    st.write(claim.get("text") or "Empty claim")
                    st.caption(
                        ", ".join(claim.get("rejection_reasons", []))
                    )

        if parse_errors:
            st.warning(f"Detected {len(parse_errors)} structured-output error(s).")
            with st.expander("Parser audit"):
                for error in parse_errors:
                    st.code(error)

        st.markdown("### 📄 Retrieved Papers")
        source_badges = {
            "both": "🟢 vector + graph", "neural": "🔵 vector", "symbolic": "🟠 graph",
            "lexical": "🟣 keyword (BM25)",
        }
        for p in papers:
            badge = source_badges.get(p["source"], p["source"])
            if p.get("lexical_support"):
                badge += " + keyword"
            with st.expander(f"{badge} {p['title'][:80]}... ({p['year']})"):
                st.write(f"**Category:** {p['category']}")
                st.write(f"**Fused score:** {round(p['score'], 3)}")
                st.write(f"**Abstract:** {(p.get('abstract') or 'N/A')[:400]}")

        # ── METRICS (see docs/EVALUATION_PROTOCOL.md) ──
        stats = provenance_stats
        semantic = result.get("semantic_support", {})
        st.markdown("### 📊 Answer Checks")
        col1, col2, col3 = st.columns(3)
        col1.metric("Citation validity",
                    f"{stats.get('valid_citations', 0)}/{stats.get('total_citations', 0)}",
                    help="Cited passage IDs that exist among the supplied evidence passages")
        col2.metric("Grounded claims",
                    f"{stats.get('grounded_claims', 0)}/{stats.get('total_claims', 0)}",
                    help="Claims whose every citation is valid; others are withheld and audited")
        if semantic.get("enabled"):
            support = semantic.get("stats", {})
            col3.metric("Evidence supports claim",
                        f"{support.get('accepted_claims', 0)}/{support.get('evaluated_claims', 0)}",
                        help="Local NLI judged the cited passages to support the claim (automated check)")
        else:
            col3.metric("Evidence supports claim", "off",
                        help="Set SEMANTIC_SUPPORT_MODEL to enable NLI checking of each claim")
        sources = {}
        for p in papers:
            sources[p["source"]] = sources.get(p["source"], 0) + 1
        st.caption("Retrieval sources: " + ", ".join(
            f"{source_badges.get(k, k)} × {v}" for k, v in sorted(sources.items())))

    # ── CONTRADICTION MODE ────────────────────────────
    elif "contradiction" in mode.lower():
        with st.spinner("Running contradiction detection pipeline..."):
            result = llm_contradict(llm_client, driver, query, top_k=top_k)

        contradictions = result.get("contradictions", [])
        st.info(f"Evaluated {len(contradictions)} contradiction candidate pairs")

        if not contradictions:
            st.warning("No contradiction candidates found for this query.")
        else:
            for i, item in enumerate(contradictions):
                p1     = item["paper1"]
                p2     = item["paper2"]
                analysis = item.get("llm_analysis", "")

                verdict = contradiction_verdict(item)
                verdict_color = "🔴" if verdict == "CONTRADICTION" else ("🟡" if verdict == "AGREEMENT" else "🔵")

                with st.expander(f"{verdict_color} Pair {i+1}: {verdict} — {p1['title'][:40]}... vs {p2['title'][:40]}..."):
                    col1, col2 = st.columns(2)
                    for column, paper, evidence in ((col1, p1, item.get("evidence_1")),
                                                    (col2, p2, item.get("evidence_2"))):
                        with column:
                            st.markdown(f"**{paper['title']}** ({paper['year']})")
                            if evidence:
                                st.info(f"“{evidence['text']}”  \n`{evidence['id']}`")
                    detail = [f"**Verdict:** {verdict}"]
                    if item.get("confidence") is not None:
                        detail.append(f"LLM confidence {item['confidence']:.2f} (uncalibrated)")
                    if item.get("nli_contradiction") is not None:
                        detail.append(f"NLI contradiction score {item['nli_contradiction']:.2f}")
                    if item.get("shared_concept_names"):
                        detail.append("shared concepts: " + ", ".join(item["shared_concept_names"][:6]))
                    st.markdown(" · ".join(detail))
                    try:
                        parsed = json.loads(analysis)
                        st.markdown(f"**Reason:** {parsed.get('reason', '')}")
                        if parsed.get("claim_1") or parsed.get("claim_2"):
                            st.markdown(f"- Paper 1 claims: {parsed.get('claim_1', '')}\n"
                                        f"- Paper 2 claims: {parsed.get('claim_2', '')}")
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        st.markdown("**LLM Analysis:**")
                        st.markdown(analysis)

    # ── HYPOTHESIS MODE ───────────────────────────────
    elif "hypothesis" in mode.lower():
        with st.spinner("Running hypothesis generation pipeline..."):
            result = llm_hypothesis(llm_client, driver, query, top_k=top_k)

        hypotheses = result.get("hypotheses", [])
        st.info(f"Generated {len(hypotheses)} research hypotheses")

        if not hypotheses:
            st.warning("No hypothesis candidates found for this query.")
        else:
            for i, item in enumerate(hypotheses):
                h = item.get("paper", {})
                llm_text = item.get("llm_hypothesis", "")

                with st.expander(f"💡 Hypothesis {i+1}: {h.get('title', '')[:60]}... ({h.get('year', '')})"):
                    st.markdown(f"**Category:** {h.get('category', 'N/A')}")
                    names = h.get("shared_concept_names") or []
                    st.markdown(f"**Shared Concepts ({h.get('shared_concepts', len(names))}):** "
                                + (", ".join(names[:10]) or "N/A"))
                    st.markdown(f"**Supporting Papers:** {h.get('supporting_papers', 'N/A')}")
                    st.markdown(f"**Evidence Score:** {h.get('evidence_score', 'N/A')}")
                    st.markdown(f"**Feasibility:** {item.get('feasibility', 'UNKNOWN')}")
                    st.markdown("**Generated Hypothesis:**")
                    st.markdown(llm_text)
                    st.markdown(f"**Missing Evidence:** {item.get('missing_evidence', 'Not provided')}")

elif run and not query:
    st.warning("Please enter a query first.")
