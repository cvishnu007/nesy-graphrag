# NeSy-GraphRAG — Automating Review and Hypothesis Generation

**Institution:** PES University, Department of Computer Science & Engineering  
**Course:** UE23CS320A/B — Capstone Project  
**Guide:** Prof. Dinesh Singh, Associate Professor  
**Team:**
- Chiyedu Vishnu — PES1UG23CS169
- Chinmay Dhar Dwivedi — PES1UG23CS165
- Dareddy Devesh Reddy — PES1UG23CS171
- Gurleen Kaur — PES1UG23CS224

**Academic Period:** Aug 2025 – Dec 2026 (Semesters 5–8)

---

## April 2026 Implementation Update (Important)

This repository now supports **two ingestion modes** behind a single switch:

- `DATA_SOURCE=arxiv` for the original ArXiv flow
- `DATA_SOURCE=s2` for Semantic Scholar Graph API flow

### What is implemented now

1. **Semantic Scholar ingestion module**: `src/ingestion/semantic_scholar_fetcher.py`
   - Uses `/paper/search/bulk` for scalable seed retrieval
   - Uses `/paper/batch` to enrich each paper with `references.paperId`
   - Enforces API pacing and retries for the 1 request/second rate limit

2. **Source switch / migration path**:
   - `src/utils/config.py` includes `DATA_SOURCE`
   - `src/ingestion/run_ingestion.py` dispatches to ArXiv or Semantic Scholar ingestion
   - File paths and Chroma collection names are source-aware

3. **Real CITES edges in Neo4j**:
   - `src/storage/neo4j_store.py` now tries to build `(:Paper)-[:CITES]->(:Paper)` from real references
   - If references are missing (or in ArXiv mode), it automatically falls back to simulated concept-overlap edges

4. **Operational hardening**:
   - `.env.example` added with all required variables and placeholders
   - model usage is now config-driven (`LLM_MODEL`) in pipeline and app

### Security requirement (must do now)

Your Semantic Scholar key was exposed in chat. **Rotate it immediately** in the Semantic Scholar dashboard and only store the new key in local `.env` (`SEMANTIC_SCHOLAR_API_KEY=...`).  
Do not commit `.env` or any real key to git.

### Exact run agenda (current)

```bash
# 0) Prepare env
cp .env.example .env
# edit .env and set:
# DATA_SOURCE=s2
# SEMANTIC_SCHOLAR_API_KEY=<rotated-key>

# 1) Ingestion (dispatches by DATA_SOURCE)
python -m src.ingestion.run_ingestion

# 2) NER
python -m src.ingestion.ner_extractor

# 3) Vector index
python -m src.storage.chroma_store

# 4) Graph load + CITES build (real refs first, fallback simulated)
python -m src.storage.neo4j_store

# 5) Pipeline smoke run
python -m src.pipeline.orchestrator

# 6) UI
streamlit run app/streamlit_app.py
```

### Validation status in this workspace (Apr 15, 2026)

- `python3 -m compileall src app` ✅ passed
- `DATA_SOURCE=s2 S2_LIMIT=20 S2_PAGE_SIZE=20 S2_BATCH_SIZE=20 python3 -m src.ingestion.run_ingestion` ✅ passed
- `DATA_SOURCE=s2 python3 -m src.ingestion.ner_extractor` ✅ passed
- `DATA_SOURCE=s2 python3 -m src.storage.chroma_store` ✅ passed (19 docs indexed in `s2_papers`)
- `DATA_SOURCE=s2 python3 -m src.storage.neo4j_store` ⛔ blocked (missing `NEO4J_URI/USERNAME/PASSWORD`)
- `DATA_SOURCE=s2 python3 -m src.pipeline.orchestrator` ⛔ blocked (missing Neo4j credentials; Groq key also required for full run)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Problem](#2-the-problem)
3. [Our Solution](#3-our-solution)
4. [Literature Survey & Research Foundation](#4-literature-survey--research-foundation)
5. [Research Gaps We Are Filling](#5-research-gaps-we-are-filling)
6. [System Architecture](#6-system-architecture)
7. [Tech Stack & Why Each Tool Was Chosen](#7-tech-stack--why-each-tool-was-chosen)
8. [Datasets](#8-datasets)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [What Has Been Done — Phase 1 and Phase 2](#10-what-has-been-done--phase-1-and-phase-2)
11. [Codebase Structure](#11-codebase-structure)
12. [Setup and Installation](#12-setup-and-installation)
13. [Running the Data Pipeline](#13-running-the-data-pipeline)
14. [Running the Application](#14-running-the-application)
15. [What Still Needs to Be Done — Phase 3 and Phase 4](#15-what-still-needs-to-be-done--phase-3-and-phase-4)
16. [Known Limitations and Constraints](#16-known-limitations-and-constraints)
17. [References](#17-references)

---

## 1. Project Overview

The volume of scientific publications is growing exponentially. Researchers today cannot manually read and synthesize everything relevant to their domain. Existing AI tools — specifically Large Language Models (LLMs) — can process text at scale but suffer from a fatal flaw in academic contexts: they hallucinate. They generate confident-sounding but factually wrong information, and they fabricate citations that do not exist.

This project builds a system that solves both problems. It automates the literature review process while guaranteeing that every single citation it produces is verified to exist in a real knowledge graph. Beyond reviewing known research, it also acts as a discovery engine — it identifies gaps in the existing literature and generates novel, data-driven research hypotheses.

The core innovation is the **Neuro-Symbolic Graph Retrieval-Augmented Generation (NeSy-GraphRAG)** architecture, which combines the semantic understanding of neural networks with the structural rigor of symbolic knowledge graphs to produce outputs that are both intelligent and provably grounded.

---

## 2. The Problem

### 2.1 The Bottleneck in Knowledge Synthesis

Scientific progress depends on researchers being able to synthesize existing knowledge. But the traditional manual literature review process is slow, biased, incomplete, and unable to scale with the rate of new publication. No single researcher can read everything relevant across all disciplines.

### 2.2 Why Current AI Tools Fail

LLMs appear to solve this — they can read and summarize text at massive scale. But they have two critical failures that make them unsuitable for academic work:

1. **Hallucination**: LLMs are probabilistic token predictors, not knowledge retrievers. They generate plausible-sounding text even when they lack factual grounding. In an academic context, a hallucinated citation is not just wrong — it is fraudulent.

2. **No Cross-Document Reasoning**: Current LLM tools retrieve isolated text chunks. They cannot reason across multiple papers simultaneously, cannot detect when two papers contradict each other, and cannot identify structural gaps in the citation network.

### 2.3 The Two-Fold Problem Statement

1. **Lack of Reliable Automation**: There is no tool that can automate literature review with verifiable citation accuracy.
2. **Passive Tools**: All existing tools only summarize known information. None use computation to actively generate novel hypotheses about unexplored research directions.

---

## 3. Our Solution

We propose a **NeSy-GraphRAG** system with three operating modes:

| Mode | What It Does |
|------|-------------|
| **Literature Review** | Retrieves the most relevant papers, validates all citations against the knowledge graph, and synthesizes a structured academic review |
| **Contradiction Detection** | Identifies pairs of papers sharing concepts but published in different years, then uses an LLM to analyze whether they genuinely contradict each other |
| **Hypothesis Generation** | Finds structural holes in the citation graph — papers sharing many concepts but never cited together — and generates novel research hypotheses |

### Why "Neuro-Symbolic"?

- **Neural component**: SPECTER embeddings + LLM (Llama 3.3 70B) handle semantic understanding, language generation, and fuzzy similarity matching.
- **Symbolic component**: Neo4j knowledge graph enforces hard constraints — a citation cannot reach the LLM unless it physically exists as a node in the graph. This eliminates hallucination at the architectural level, not the prompt level.

---

## 4. Literature Survey & Research Foundation

Ten papers directly informed our design decisions.

### 4.1 Automating Research Synthesis (Susnjak et al., 2024)
Proposed scaling Systematic Literature Reviews using domain-fine-tuned LLMs with provenance tracking and special token insertion to tie outputs to source passages. Validated our core goal of traceable, citation-grounded synthesis.

### 4.2 Neurosymbolic AI for Attribution (Tilwani et al., 2024) — IEEE Intelligent Systems
Demonstrated that combining neural networks with symbolic knowledge graphs can enforce source provenance and drastically reduce fabricated citations. This is the direct academic precedent for our NeSy architecture.

### 4.3 Contradiction Detection — DisContNet (Kutty et al., 2023)
Fine-tuned BERT and DistilBERT on a composite dataset with four contradiction labels (General, Negation-induced, Numerical, Non-contradictory). Found that higher parameter count is critical for complex inferential contradictions. Limitation: only handles sentence-level, not cross-document contradictions — a gap our system addresses.

### 4.4 Deep Learning for Hypothesis Generation — SciHypo (Tadiparthi et al., 2024)
Built a pipeline for generating scientific hypotheses using entity-relationship extraction and transformer-based generation. Limitation: no formal logical structure, output quality depends heavily on input quality.

### 4.5 Controllable Logical Hypothesis Generation — CtrlHGen (Gao et al., 2025)
Introduced structured hypothesis generation using abductive reasoning over knowledge graphs with two-stage training (supervised learning + reinforcement learning with semantic rewards). Directly inspired our hypothesis generation module.

### 4.6 Causal Hypothesis Generation with LLMs (Cohrs et al., 2025)
Formalizes LLMs as probabilistic imperfect experts for causal hypothesis generation. Proposes hybrid methods combining LLM-generated priors with constraint-based causal discovery algorithms.

### 4.7 Scientific Hypothesis Generation Survey (Kulkarni et al., 2025)
Classifies hypothesis generation into 9 categories and validation into 10 strategies. Identifies low novelty, weak feasibility checks, and poor interpretability as major challenges. Proposes novelty-aware training as future direction.

### 4.8 Survey of AI for Systematic Literature Reviews (Torre-López et al., 2023)
First comprehensive survey of AI for SLR automation across 9,000+ references. Found that SVMs and Naive Bayes still dominate; full automation is not achievable; the field is immature and fragmented.

### 4.9 Agent-Based Auto Research (Liu et al., 2025)
The most complete vision for end-to-end LLM-agent research automation spanning 8 modules (Literature → Idea → Method → Experiment → Paper → Evaluation → Rebuttal → Promotion). Our system focuses on the Literature and Hypothesis modules.

### 4.10 Agentic AI for Scientific Discovery (Gridach et al., 2025 — ICLR)
The definitive 2025 survey of agentic AI for scientific discovery. Key finding: literature review is the weakest phase for almost all current agentic systems. Explicitly recommends neurosymbolic enhancements — which is exactly what we build.

---

## 5. Research Gaps We Are Filling

| Gap | Our Solution |
|-----|-------------|
| No end-to-end system unifying retrieval, reasoning, contradiction checks, and hypothesis generation | Our four-stage NeSy-GraphRAG pipeline covers all of these in one coherent workflow |
| No provenance-aware synthesis — generated insights don't map back to source papers | Every paper ID is validated against Neo4j before reaching the LLM. If it doesn't exist in the graph, it cannot be cited |
| Limited contradiction detection — existing models only handle sentence-level, not cross-document | Our graph-based contradiction engine compares papers across years sharing semantic concepts |
| No standard benchmarks for SLR automation and hypothesis generation | We propose three custom metrics (TS, RDI, HNS) as a starting point for standardized evaluation |

---

## 6. System Architecture

### Layered Architecture

```
+---------------------------------------------------------+
|                   FRONTEND LAYER                        |
|          Streamlit UI + PyVis graph visualization       |
+-------------------------+-------------------------------+
                          |
+-------------------------v-------------------------------+
|              NEUROSYMBOLIC LOGIC LAYER                  |
|                                                         |
|  +-------------+  +-----------+  +------------------+  |
|  |  GraphRAG   |  | Validator |  | Discovery Engine |  |
|  |  Retrieval  |  | (blocks   |  | (link prediction |  |
|  |  Engine     |  | halluc.)  |  | for hypotheses)  |  |
|  +-------------+  +-----------+  +------------------+  |
+-------------------------+-------------------------------+
                          |
+-------------------------v-------------------------------+
|                 HYBRID STORAGE LAYER                    |
|                                                         |
|  +----------------------+  +--------------------------+ |
|  |   ChromaDB           |  |   Neo4j AuraDB           | |
|  |   SPECTER embeddings |  |   Paper/Author/Concept   | |
|  |   Semantic search    |  |   nodes + CITES edges    | |
|  +----------------------+  +--------------------------+ |
+---------------------------------------------------------+
```

### The Four-Stage Pipeline

Every query passes through all four stages:

**Stage 1 — Neural Retrieval**  
The user's query is encoded into a 768-dimensional vector using SPECTER 2.0. ChromaDB performs cosine similarity search against the 10,000 stored paper embeddings and returns the top 10 most semantically similar papers.

**Stage 2 — Symbolic Expansion**  
The IDs of the top 10 papers are sent to Neo4j. The graph is traversed 1–2 hops along CITES edges to surface related, highly-connected papers that semantic search may have missed — papers that are conceptually adjacent but use different terminology.

**Stage 3 — Citation Validation**  
Every paper ID from stages 1 and 2 is verified against the Neo4j graph with a direct node lookup. Any ID not corresponding to a real node is dropped before the LLM ever sees it. This is the hallucination firewall — it operates at the architectural level, not the prompt level.

**Stage 4 — LLM Synthesis**  
Verified papers are formatted into a pipe-separated TOON (Title|Year|Category|Abstract) context and passed to Llama 3.3 70B via Groq. The LLM synthesizes, analyzes, or generates hypotheses depending on the selected mode.

---

## 7. Tech Stack & Why Each Tool Was Chosen

| Component | Tool Chosen | Why | Rejected Alternative |
|-----------|-------------|-----|----------------------|
| Orchestration | LlamaIndex | Purpose-built for RAG; native Property Graph support; minimal boilerplate | LangChain — more general-purpose, heavier, no native graph support |
| Graph Database | Neo4j AuraDB Free | Built for graph scaling; includes GDS library with Link Prediction algorithms | NetworkX — runs entirely in RAM, would crash on 10,000+ nodes |
| Embedding Model | SPECTER 2.0 (allenai-specter) | Pre-trained on scientific papers; handles domain jargon; 768-dim vectors | OpenAI ada-002 — not trained on scientific data; mishandles academic terminology |
| Vector Database | ChromaDB | Zero-latency local prototyping; persistent client; cosine similarity built-in | Pinecone — better for production cloud deployment; planned for Phase 3 |
| LLM | Llama 3.3 70B via Groq | Open weights; Groq hardware gives fast inference; sufficient for synthesis | GPT-4o — proprietary, expensive at scale, no control over model updates |
| NLP/NER | spaCy en_core_web_sm | Fast; CPU-friendly; good noun chunk extraction | NLTK — lower accuracy, older API |
| Frontend | Streamlit + PyVis | Pure Python; rapid prototyping; PyVis renders interactive citation graphs | Flask/React — requires separate frontend build, overkill for academic prototype |

### Rejected Architecture Approaches

1. **Standard Vector-Only RAG**: Cannot traverse citation relationships or find structural holes. Weak cross-document reasoning.
2. **Long-Context LLMs**: Suffers from "Lost-in-the-Middle" at very long contexts. Cannot scale to millions of papers.
3. **Pure Agentic Systems**: Non-deterministic, unstable, cannot guarantee provenance. Not suitable for academic integrity requirements.

---

## 8. Datasets

### Phase 1 — Explored

**Cornell University arXiv Dataset (via Kaggle)**  
1.7 million scholarly articles across CS, Physics, Mathematics, Statistics. Used for NLP training and semantic analysis.

**Scientific Papers Dataset (Hugging Face)**  
~348,000 records: 215,000 arXiv + 133,000 PubMed. Unique value: paired full-text + abstract. Used for summarization benchmarking.

**S2ORC — Semantic Scholar Open Research Corpus**  
136 million papers, 12 million with full text. Advanced PDF-to-JSON parsing preserving section hierarchies and inline citations. Key asset for Knowledge Graph construction.

### Phase 2 — Used in Prototype

- **Size**: 10,000 Computer Science papers
- **Source**: ArXiv official API
- **Composition**: 2,000 papers per year × 5 years (2020–2024)
- **Categories**: All 40 CS sub-categories
- **Graph result**: ~9,990 Paper nodes, 24,176 CITES edges in Neo4j

---

## 9. Evaluation Metrics

Three custom metrics were proposed in Phase 1. These do not yet exist as standard benchmarks — we are defining them.

### Trustworthiness Score (TS)

```
TS = 0.5 × Citation Integrity + 0.5 × (1 - Hallucination Rate)
```

Measures how safe and grounded the system output is. Target: TS > 0.90.

### Reasoning Depth Index (RDI)

```
RDI = 0.5 × Cross-Document Support + 0.5 × Contradiction Resolution Rate
```

Measures whether the AI is synthesizing complex information or just retrieving text. Target: RDI > 0.75.

### Hypothesis Novelty Score (HNS)

```
HNS = Average Graph Distance between connected concept nodes
```

Higher graph distance = more novel connection. A hypothesis bridging two distant sub-fields scores higher than one connecting adjacent, heavily-cited concepts. Target: HNS > 3.0 average hop distance.

---

## 10. What Has Been Done — Phase 1 and Phase 2

### Phase 1 — Completed (Semester 5, Aug–Dec 2025)

- Defined the problem: information overload, LLM hallucination, passive tools
- Conducted a 10-paper literature survey establishing academic foundation
- Identified 4 research gaps that existing systems do not address
- Proposed the NeSy-GraphRAG architecture
- Selected datasets (arXiv, Hugging Face Scientific Papers, S2ORC)
- Defined evaluation metrics (TS, RDI, HNS)
- Submitted Phase 1 dissertation report

### Phase 2 — Completed (Semester 6)

Everything below was built and tested in `capstone.ipynb` on Google Colab with Google Drive storage, then refactored into the modular Python package.

**Data Ingestion and Cleaning**
- Fetched 10,000 CS papers from ArXiv API across all 40 CS categories, 2,000 per year for 2020–2024
- Applied text cleaning: removed LaTeX math (`$...$`, `$$...$$`), LaTeX commands (`\command{}`), URLs, special characters, converted to lowercase
- Dropped papers with missing titles/abstracts or abstracts under 30 words
- Saved to `arxiv_raw.json` (raw) and `arxiv_clean.json` (cleaned)

**Semantic Indexing**
- Loaded SPECTER 2.0 (`allenai-specter`) via sentence-transformers
- Encoded all 9,990 cleaned abstracts into 768-dimensional vectors in batches of 64
- Stored in ChromaDB persistent client with cosine similarity metric
- Implemented resume support — already-stored papers are skipped on re-run

**Named Entity Recognition**
- Ran spaCy `en_core_web_sm` on each abstract (truncated to 1,000 chars for speed)
- Extracted entities with labels: ORG, PRODUCT, GPE, WORK_OF_ART, EVENT
- Extracted noun chunks up to 4 words
- Applied rule-based noise filter: removed common stopwords, single words under 4 characters, and phrases starting with articles
- Saved enriched data to `arxiv_ner.json`

**Knowledge Graph Construction (Neo4j)**
- Created three node types: Paper (id, title, year, category, abstract), Author (name), Concept (name)
- Created two relationship types: AUTHORED_BY (Paper→Author), RELATED_TO (Paper→Concept)
- Inserted in batches of 500 using UNWIND Cypher for performance
- Created CITES edges: two papers are linked if they share 2 or more concepts (p1.id < p2.id to avoid duplicate edges)
- Result: 9,990 Paper nodes, 24,176 CITES edges

**NeSy Retrieval Pipeline**
- `neural_retrieve()`: SPECTER encodes the query, ChromaDB returns top-k papers by cosine similarity
- `symbolic_expand()`: Neo4j traverses 1–2 CITES hops from retrieved papers, returns highly-connected neighbors not in original results
- `nesy_retrieve()`: Merges both; papers appearing in both get score boost; returns re-ranked top-k

**Citation Validator**
- `validate_citations()`: Takes paper IDs, runs Neo4j MATCH, returns only IDs that exist as real nodes
- Prints count of verified vs. blocked citations for every query
- Any paper ID not found in Neo4j is silently dropped before reaching the LLM

**Contradiction Detection**
- `detect_contradictions()`: Finds paper pairs from the retrieved set that share 2 or more concepts AND were published in different years
- `llm_contradict()`: Sends each candidate pair to Llama 3.3 70B requesting VERDICT + REASON + CLAIM 1 + CLAIM 2

**Hypothesis Generation**
- `generate_hypotheses()`: Neo4j query finds papers sharing 2 or more concepts with the query papers but with no existing CITES edge — structural holes
- `llm_hypothesis()`: Sends each structural hole to Llama 3.3 70B with query context, generates HYPOTHESIS + RATIONALE + POTENTIAL IMPACT

**LLM Integration**
- Connected Groq API using `llama-3.3-70b-versatile` (replaced decommissioned older model)
- Three separate prompt templates for review, contradiction, and hypothesis modes
- Papers formatted as pipe-separated TOON context (title|year|category|abstract)

**Streamlit Application**
- Full UI with sidebar mode selector and adjustable top-k slider
- Source badges on results: neural+symbolic (green), neural only (blue), symbolic only (orange)
- Contradiction results show verdict color coding: red (contradiction), yellow (agreement), blue (different scope)
- Originally deployed via Google Colab + localtunnel

**Repository Migration**
- Refactored from single monolithic notebook into modular Python package
- Separated into ingestion, storage, pipeline, and app layers
- All credentials moved to `.env` file, never hardcoded
- All hardcoded Google Drive paths replaced with config.py constants
- Each module independently runnable via `python -m src.<module>`

---

## 11. Codebase Structure

```
nesy-graphrag/
|
+-- .env                              # ALL credentials -- never commit this
+-- .env.example                      # Safe template; copy to .env
+-- .gitignore                        # Excludes .env, venv/, data/, __pycache__/
+-- requirements.txt                  # All pip dependencies
+-- README.md                         # This file
|
+-- data/                             # Generated by pipeline -- gitignored
|   +-- arxiv_raw.json                # Raw ArXiv fetch output
|   +-- arxiv_clean.json              # After text cleaning + dedup
|   +-- arxiv_ner.json                # After spaCy NER extraction
|   +-- s2_raw.json                   # Raw normalized Semantic Scholar output
|   +-- s2_clean.json                 # Cleaned Semantic Scholar records
|   +-- s2_ner.json                   # Semantic Scholar + extracted concepts
|   +-- chromadb/                     # ChromaDB persistent vector index
|
+-- src/
|   +-- utils/
|   |   +-- config.py                 # Reads .env -- single source of truth for all settings
|   |
|   +-- ingestion/
|   |   +-- arxiv_fetcher.py          # Fetch + clean 10k papers from ArXiv API
|   |   +-- semantic_scholar_fetcher.py  # Fetch + enrich papers from S2 Graph API
|   |   +-- run_ingestion.py          # Dispatch by DATA_SOURCE (arxiv|s2)
|   |   +-- ner_extractor.py          # spaCy NER + noise filter
|   |
|   +-- storage/
|   |   +-- chroma_store.py           # SPECTER encode + ChromaDB index + query (source-aware collection)
|   |   +-- neo4j_store.py            # Neo4j: insert papers + real/fallback CITES creation
|   |
|   +-- pipeline/
|       +-- retrieval.py              # neural_retrieve + symbolic_expand + nesy_retrieve
|       +-- validator.py              # validate_citations -- hallucination firewall
|       +-- review.py                 # llm_review -- literature review mode
|       +-- contradiction.py          # detect_contradictions + llm_contradict
|       +-- hypothesis.py             # generate_hypotheses + llm_hypothesis
|       +-- orchestrator.py           # graphrag_query() -- master entry point
|
+-- app/
|   +-- streamlit_app.py              # Full Streamlit UI
```

### Data Flow

```
DATA_SOURCE (arxiv | s2)
    |
    v
run_ingestion.py  -->  arxiv_fetcher.py OR semantic_scholar_fetcher.py
                  -->  data/<source>_raw.json
                  -->  data/<source>_clean.json
    |
    v
ner_extractor.py  -->  data/<source>_ner.json
    |
    v
chroma_store.py   -->  data/chromadb/         (768-dim SPECTER vectors, source-aware collection)
neo4j_store.py    -->  Neo4j AuraDB           (Paper + Author + Concept + CITES)
                  -->  CITES from real references when available, fallback to concept overlap
    |
    v
retrieval.py      <--  ChromaDB (neural)
                  <--  Neo4j (symbolic expansion)
    |
    v
validator.py      <--  Neo4j (citation existence check)
    |
    v
review.py / contradiction.py / hypothesis.py
                  -->  Groq API (Llama 3.3 70B)
    |
    v
orchestrator.py   -->  structured result dict
    |
    v
streamlit_app.py  -->  UI
```

---

## 12. Setup and Installation

### Prerequisites

- Python 3.10 or higher
- A Neo4j AuraDB Free account: https://neo4j.com/cloud/platform/aura-graph-database/
- A Groq API key: https://console.groq.com

### Step 1: Clone and enter the repo

```bash
git clone <your-repo-url>
cd nesy-graphrag
```

### Step 2: Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 4: Configure credentials

Create `.env` from template and fill in your values:

```bash
cp .env.example .env
```

Minimum required values:

```
DATA_SOURCE=s2
NEO4J_URI=neo4j+s://YOUR_INSTANCE.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
GROQ_API_KEY=gsk_your_key_here
SEMANTIC_SCHOLAR_API_KEY=your_rotated_key_here
EMBEDDING_MODEL=allenai-specter
LLM_MODEL=llama-3.3-70b-versatile
```

The `.env` file is gitignored. Never commit it.

### Step 5: Create the data directory

```bash
mkdir -p data/chromadb
```

---

## 13. Running the Data Pipeline

Run these in order. Each step produces a file that the next step reads. This is a one-time setup.

### Step 1: Ingest and clean papers (source-aware)

```bash
python -m src.ingestion.run_ingestion
```

Behavior:
- If `DATA_SOURCE=arxiv`: fetches ArXiv data and writes `data/arxiv_raw.json`, `data/arxiv_clean.json`
- If `DATA_SOURCE=s2`: fetches Semantic Scholar data (`/paper/search/bulk` + `/paper/batch`) and writes `data/s2_raw.json`, `data/s2_clean.json`

The S2 path enriches each paper with references to support real CITES edges.

### Step 2: Run NER extraction (~10–15 minutes)

```bash
python -m src.ingestion.ner_extractor
```

Reads `CLEAN_FILE` and writes `NER_FILE` based on `DATA_SOURCE`.

### Step 3: Build ChromaDB index (~20–30 minutes)

```bash
python -m src.storage.chroma_store
```

Builds/updates a source-aware Chroma collection (e.g., `arxiv_papers` or `s2_papers`) in `data/chromadb/`.

### Step 4: Populate Neo4j (~5–10 minutes)

```bash
python -m src.storage.neo4j_store
```

Reads `NER_FILE`, inserts Paper/Author/Concept nodes, then builds CITES edges:
- real reference-based CITES when `USE_REAL_CITATIONS=true` and references exist
- fallback concept-overlap CITES otherwise

### Step 5: Test the full pipeline

```bash
python -m src.pipeline.orchestrator
```

Runs all three modes against a test query. If this passes, the entire system is working.

---

## 14. Running the Application

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`. Select mode in the sidebar, enter a research query, set top-k, click Run Query.

---

## 15. What Still Needs to Be Done — Phase 3 and Phase 4

### Phase 3 — Full Implementation (Semester 7)

**Evaluation Framework**
- Implement TS, RDI, and HNS as automated scoring functions
- Build a benchmark test suite with ground-truth queries and expected outputs
- Run comparison against standard vector RAG baseline to measure hallucination reduction quantitatively
- Document and report results

**PDF Ingestion Pipeline**
- Build a PDF parser using S2ORC-style extraction
- Handle multi-column layouts, LaTeX equations, and figure captions
- Add document upload to Streamlit UI so users can add their own papers to the index

**Scale-Up**
- Migrate ChromaDB to Pinecone for cloud deployment
- Expand from 10,000 to 100,000+ papers using S2ORC
- Optimize Cypher queries for larger graph performance

**Contradiction Detection Improvement**
- Replace structural proxy (shared concepts + different years) with a proper NLI model
- Candidate: fine-tuned BERT on a contradiction dataset, following the DisContNet approach from the literature survey
- Evaluate with the RDI metric

**Hypothesis Validation**
- Add a feasibility check step that cross-references generated hypotheses against the literature
- Implement the HNS metric to score and rank hypotheses by novelty
- Filter out low-novelty hypotheses automatically

**Graph Visualization**
- Integrate PyVis citation graph rendering into the Streamlit UI
- Allow interactive exploration of the knowledge graph

**Testing**
- Write unit tests for all modules in `tests/`
- Cover: empty queries, no ChromaDB results, Neo4j connection failures
- Test the citation validator with deliberately fabricated paper IDs

### Phase 4 — Publication and Patenting (Semester 8)

- Write up quantitative results for academic publication
- Target venues: EMNLP, ACL, or ICLR workshops on Scientific NLP
- Document the NeSy-GraphRAG architecture as a reproducible system paper
- File a patent application for the citation-validation-as-hallucination-firewall approach, pending IP review
- Open-source the codebase after IP review

---

## 16. Known Limitations and Constraints

**Latency**: Each query involves two external service calls (ChromaDB + Neo4j) plus an LLM call. Typical response time is 5–10 seconds. The 2-hop graph traversal must be strictly limited for real-time use.

**Citation Edge Quality Depends on Source**:  
- `DATA_SOURCE=s2` can now build real CITES edges from Semantic Scholar references.  
- `DATA_SOURCE=arxiv` (or sparse references) falls back to simulated concept-overlap CITES edges.

**PDF Noise**: The prototype uses pre-cleaned ArXiv abstracts. Real-world PDF parsing is significantly noisier. The pre-processing layer is a known weak point.

**NER Quality**: spaCy `en_core_web_sm` is a small model. It struggles with highly technical academic terms. A domain-specific NER model trained on scientific text would improve concept extraction quality.

**10,000 Paper Limit**: At this scale, the graph is sparse and hypothesis generation finds fewer structural holes than a larger graph would. Phase 3 targets 100,000+ papers.

**ChromaDB Scaling**: The local persistent client is suitable for prototyping but not for concurrent users or datasets much larger than ~100,000 vectors. Pinecone migration is required before any public deployment.

---

## 17. References

[1] T. Susnjak et al., "Automating research synthesis with domain-specific large language model fine-tuning," arXiv:2404.08680, 2024.

[2] C. Liu et al., "A vision for auto research with LLM agents," arXiv:2504.18765, 2025.

[3] J. de la Torre-López, A. Ramírez, and J. R. Romero, "Artificial intelligence to automate the systematic review of scientific literature," Computing, vol. 105, no. 10, pp. 2171–2194, Oct. 2023.

[4] M. Tadiparthi et al., "SCIHYPO — A deep learning framework for data-driven scientific hypothesis generation," ICOECA 2024, pp. 1037–1042.

[5] M. Gridach et al., "Agentic AI for scientific discovery: A survey of progress, challenges, and future directions," ICLR 2025.

[6] D. Tilwani, R. Venkataramanan, and A. P. Sheth, "Neurosymbolic AI approach to attribution in large language models," IEEE Intelligent Systems, 2024.

[7] Y. Gao et al., "Controllable logical hypothesis generation for abductive reasoning in knowledge graphs," arXiv:2505.20948, 2025.

[8] R. J. Kutty, R. P. N, and S. S. Adiga, "DisContNet: Contradiction detection in texts using transformers," ICSTCEE 2023, Bengaluru.

[9] K.-H. Cohrs et al., "Large language models for causal hypothesis generation in science," Mach. Learn.: Sci. Technol., vol. 6, Art. no. 013001, Jan. 2025.

[10] A. Kulkarni et al., "Scientific hypothesis generation and validation: Methods, datasets, and future directions," arXiv:2505.04651, 2025.
