import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline.orchestrator import get_neo4j, get_groq
from src.pipeline.review import llm_review
from src.pipeline.contradiction import llm_contradict
from src.pipeline.hypothesis import llm_hypothesis
from src.pipeline.verdicts import extract_verdict
from src.storage.chroma_store import get_collection
from src.utils.config import CHROMA_COLLECTION, DATA_SOURCE, LLM_MODEL
from src.pipeline.metrics import compute_all_metrics

# ════════════════════════════════════════════════════
# INIT (cached so it only runs once)
# ════════════════════════════════════════════════════
@st.cache_resource
def load_resources():
    driver      = get_neo4j()
    groq_client = get_groq()
    collection  = get_collection()
    return driver, groq_client, collection

driver, groq_client, collection = load_resources()


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
    top_k = st.slider("Papers to retrieve", min_value=3, max_value=15, value=10)
    st.markdown("---")
    st.markdown("**Pipeline:**")
    paper_nodes, cites_edges = graph_stats(driver)
    st.markdown(f"- Source: `{DATA_SOURCE}`")
    st.markdown(f"- Chroma collection: `{CHROMA_COLLECTION}` ({collection.count()} vectors)")
    st.markdown(f"- Neo4j: {paper_nodes} papers, {cites_edges} CITES edges")
    st.markdown(f"- LLM: `{LLM_MODEL}`")

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
            result = llm_review(groq_client, driver, query, top_k=top_k)

        papers   = result["papers"]
        answer   = result["answer"]
        verified = result["verified"]

        st.success(f"Retrieved {len(papers)} papers — {len(verified)}/{len(papers)} citations verified")

        st.markdown("### 📝 Literature Review")
        st.markdown(answer)

        st.markdown("### 📄 Retrieved Papers")
        for p in papers:
            badge = "🟢 neural+symbolic" if p["source"] == "both" else ("🔵 neural" if p["source"] == "neural" else "🟠 symbolic")
            with st.expander(f"{badge} {p['title'][:80]}... ({p['year']})"):
                st.write(f"**Category:** {p['category']}")
                st.write(f"**Score:** {round(p['score'], 3)}")
                st.write(f"**Abstract:** {(p.get('abstract') or 'N/A')[:400]}")

        # ── METRICS ──
        metrics = compute_all_metrics(
            result,
            driver=driver,
        )
        st.markdown("### 📊 Evaluation Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("TS (Trustworthiness)", f"{metrics['ts']['ts']:.3f}",
                    help="Target ≥ 0.90")
        col2.metric("NBR (NeSy Boost)", f"{metrics['nbr']['nbr']:.3f}",
                    help="Target > 0.30 — proves graph adds value")
        col3.metric("ATD (Temporal Range)", f"{metrics['atd']['atd']:.3f}",
                    help="1.0 = all 5 years represented")
        col4.metric("RDI (Reasoning Depth)", f"{metrics['rdi']['rdi']:.3f}",
                    help="Target ≥ 0.75")

        # ── Flag missing years explicitly ──
        missing_years = metrics["atd"].get("missing_years", [])
        if missing_years:
            st.warning(
                f"⚠️ **Missing years in results:** {', '.join(str(y) for y in missing_years)}. "
                f"Papers from these years were not found in the retrieval results, "
                f"which may indicate gaps in dataset coverage or query specificity."
            )

    # ── CONTRADICTION MODE ────────────────────────────
    elif "contradiction" in mode.lower():
        with st.spinner("Running contradiction detection pipeline..."):
            result = llm_contradict(groq_client, driver, query, top_k=5)

        contradictions = result.get("contradictions", [])
        st.info(f"Found and verified {len(contradictions)} contradiction candidate pairs")

        if not contradictions:
            st.warning("No contradiction candidates found for this query.")
        else:
            for i, item in enumerate(contradictions):
                p1     = item["paper1"]
                p2     = item["paper2"]
                analysis = item.get("llm_analysis", "")

                verdict = extract_verdict(analysis)
                verdict_color = "🔴" if verdict == "CONTRADICTION" else ("🟡" if verdict == "AGREEMENT" else "🔵")

                with st.expander(f"{verdict_color} Pair {i+1}: {p1['title'][:40]}... vs {p2['title'][:40]}..."):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Paper 1 ({p1['year']})**")
                        st.write(p1['title'])
                    with col2:
                        st.markdown(f"**Paper 2 ({p2['year']})**")
                        st.write(p2['title'])
                    st.markdown("**LLM Analysis:**")
                    st.markdown(analysis)

    # ── HYPOTHESIS MODE ───────────────────────────────
    elif "hypothesis" in mode.lower():
        with st.spinner("Running hypothesis generation pipeline..."):
            result = llm_hypothesis(groq_client, driver, query, top_k=5)

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
                    st.markdown(f"**Shared Concepts:** {h.get('shared_concepts', 'N/A')}")
                    st.markdown("**Generated Hypothesis:**")
                    st.markdown(llm_text)

elif run and not query:
    st.warning("Please enter a query first.")
