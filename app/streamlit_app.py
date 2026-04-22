import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline.orchestrator import get_neo4j, get_groq
from src.pipeline.retrieval import nesy_retrieve
from src.pipeline.validator import validate_citations
from src.pipeline.contradiction import detect_contradictions
from src.pipeline.hypothesis import generate_hypotheses
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
        with st.spinner("Retrieving papers..."):
            papers   = nesy_retrieve(driver, query, top_k=top_k)
            verified = validate_citations(driver, [p["id"] for p in papers if p.get("id")])

        st.success(f"Retrieved {len(papers)} papers — {len(verified)}/{len(papers)} citations verified")

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

        with st.spinner("LLM generating answer..."):
            response = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3
            )
            answer = response.choices[0].message.content

        st.markdown("### 📝 Literature Review")
        st.markdown(answer)

        st.markdown("### 📄 Retrieved Papers")
        for p in papers:
            badge = "🟢 neural+symbolic" if p["source"] == "both" else ("🔵 neural" if p["source"] == "neural" else "🟠 symbolic")
            with st.expander(f"{badge} {p['title'][:80]}... ({p['year']})"):
                st.write(f"**Category:** {p['category']}")
                st.write(f"**Score:** {round(p['score'], 3)}")
                st.write(f"**Abstract:** {(p.get('abstract') or 'N/A')[:400]}")

        # ── METRICS — indented inside the review block ──
        metrics = compute_all_metrics(
            {"papers": papers, "answer": answer, "verified": verified}
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

    # ── CONTRADICTION MODE ────────────────────────────
    elif "contradiction" in mode.lower():
        with st.spinner("Finding contradiction candidates..."):
            contradictions = detect_contradictions(driver, query, top_k=5)

        st.info(f"Found {len(contradictions)} candidate pairs — verifying with LLM...")

        if not contradictions:
            st.warning("No contradiction candidates found for this query.")
        else:
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

                with st.spinner(f"Checking pair {i+1}/{len(contradictions)}..."):
                    response = groq_client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,
                        temperature=0.3
                    )
                    result = response.choices[0].message.content

                verdict_color = "🔴" if "CONTRADICTION" in result else ("🟡" if "AGREEMENT" in result else "🔵")

                with st.expander(f"{verdict_color} Pair {i+1}: {p1['title'][:40]}... vs {p2['title'][:40]}..."):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Paper 1 ({p1['year']})**")
                        st.write(p1['title'])
                    with col2:
                        st.markdown(f"**Paper 2 ({p2['year']})**")
                        st.write(p2['title'])
                    st.markdown("**LLM Analysis:**")
                    st.markdown(result)

    # ── HYPOTHESIS MODE ───────────────────────────────
    elif "hypothesis" in mode.lower():
        from src.pipeline.retrieval import neural_retrieve
        with st.spinner("Finding structural holes in knowledge graph..."):
            hypotheses   = generate_hypotheses(driver, query, top_k=5)
            query_papers = neural_retrieve(query, top_k=3)

        st.info(f"Found {len(hypotheses)} structural holes — generating hypotheses with LLM...")

        if not hypotheses:
            st.warning("No hypothesis candidates found for this query.")
        else:
            query_context = "\n".join([
                f"- {p['title'][:80]} ({p['year']}): {(p.get('abstract') or '')[:200]}"
                for p in query_papers
            ])

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

                with st.spinner(f"Generating hypothesis {i+1}/{len(hypotheses)}..."):
                    response = groq_client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,
                        temperature=0.3
                    )
                    result = response.choices[0].message.content

                with st.expander(f"💡 Hypothesis {i+1}: {h['title'][:60]}... ({h['year']})"):
                    st.markdown(f"**Category:** {h['category']}")
                    st.markdown(f"**Shared Concepts:** {h['shared_concepts']}")
                    st.markdown("**Generated Hypothesis:**")
                    st.markdown(result)

elif run and not query:
    st.warning("Please enter a query first.")