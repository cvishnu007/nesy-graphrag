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
**Current Status:** Prototyping — Phase 2 Complete / Phase 3 In Progress
**Last Validated:** April 22, 2026

---

## ⚠️ Prototype Notice

This system is in active prototyping. The full end-to-end pipeline runs successfully on a 10,000-paper ArXiv dataset. The Semantic Scholar ingestion path is implemented but requires credential setup. Several evaluation metrics have known limitations documented in this file. Nothing here should be treated as production-ready.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Problem](#2-the-problem)
3. [Our Solution](#3-our-solution)
4. [System Architecture](#4-system-architecture)
5. [Data Flow](#5-data-flow)
6. [The Four-Stage Pipeline](#6-the-four-stage-pipeline)
7. [Tech Stack](#7-tech-stack)
8. [Codebase Structure](#8-codebase-structure)
9. [What Has Been Done](#9-what-has-been-done)
10. [Evaluation Metrics — Implemented](#10-evaluation-metrics--implemented)
11. [Live Pipeline Output Analysis](#11-live-pipeline-output-analysis)
12. [Setup and Installation](#12-setup-and-installation)
13. [Running the Pipeline](#13-running-the-pipeline)
14. [What Is Pending](#14-what-is-pending)
15. [Known Bugs and Limitations](#15-known-bugs-and-limitations)
16. [What Can Be Improved](#16-what-can-be-improved)
17. [References](#17-references)

---

## 1. Project Overview

The volume of scientific publications is growing exponentially. Researchers cannot manually read and synthesize everything relevant to their domain. Existing AI tools suffer from a fatal flaw in academic contexts: they hallucinate. They generate confident-sounding but factually wrong citations that do not exist.

This project builds a system that solves both problems simultaneously. It automates literature review while guaranteeing that every citation it produces is verified against a real knowledge graph. Beyond reviewing known research, it acts as a discovery engine — identifying gaps in the literature and generating novel, data-driven research hypotheses.

The core innovation is the **Neuro-Symbolic Graph Retrieval-Augmented Generation (NeSy-GraphRAG)** architecture, which combines the semantic understanding of neural networks with the structural rigour of symbolic knowledge graphs.

---

## 2. The Problem

### 2.1 The Bottleneck in Knowledge Synthesis

Scientific progress depends on researchers synthesising existing knowledge. The manual literature review process is slow, biased, incomplete, and unable to scale with the rate of new publication.

### 2.2 Why Current AI Tools Fail

1. **Hallucination** — LLMs are probabilistic token predictors. They generate plausible-sounding text even when they lack factual grounding. A hallucinated citation in an academic paper is not just wrong — it is fraudulent.
2. **No Cross-Document Reasoning** — Current tools retrieve isolated text chunks. They cannot reason across multiple papers, cannot detect when two papers contradict each other, and cannot identify structural gaps in a citation network.

---

## 3. Our Solution

A NeSy-GraphRAG system with three operating modes:

| Mode | What It Does |
|------|-------------|
| **Literature Review** | Retrieves relevant papers, validates all citations against the knowledge graph, synthesises a structured academic review |
| **Contradiction Detection** | Finds paper pairs sharing concepts from different years, uses LLM to determine if they genuinely contradict each other |
| **Hypothesis Generation** | Finds structural holes in the citation graph — papers sharing concepts but never cited together — and generates novel research hypotheses |

### Why Neuro-Symbolic?

- **Neural** — SPECTER embeddings + Llama 3.3 70B handle semantic understanding and language generation
- **Symbolic** — Neo4j enforces hard constraints. A citation cannot reach the LLM unless it physically exists as a verified node in the graph. This eliminates hallucination at the **architectural level**, not the prompt level.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND LAYER                           │
│              Streamlit UI  ·  PyVis graph (planned)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                   NEUROSYMBOLIC LOGIC LAYER                     │
│                                                                 │
│  ┌──────────────┐   ┌─────────────┐   ┌──────────────────────┐ │
│  │   NeSy       │   │  Citation   │   │   Discovery Engine   │ │
│  │  Retrieval   │──▶│  Validator  │──▶│  (Hypothesis / Cont) │ │
│  │  Engine      │   │  (firewall) │   │                      │ │
│  └──────────────┘   └─────────────┘   └──────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 METRICS LAYER (new)                      │   │
│  │         TS · NBR · ATD · RDI  (src/pipeline/metrics.py) │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    HYBRID STORAGE LAYER                         │
│                                                                 │
│  ┌───────────────────────┐    ┌───────────────────────────────┐ │
│  │  ChromaDB             │    │  Neo4j AuraDB                 │ │
│  │  SPECTER embeddings   │    │  Paper · Author · Concept     │ │
│  │  Cosine similarity    │    │  CITES · AUTHORED_BY          │ │
│  │  ~10k vectors         │    │  RELATED_TO edges             │ │
│  └───────────────────────┘    └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     INGESTION LAYER                             │
│                                                                 │
│   ArXiv API fetcher          Semantic Scholar Graph API         │
│   (arxiv_fetcher.py)         (semantic_scholar_fetcher.py)      │
│         │                            │                         │
│         └──────────┬─────────────────┘                         │
│                    ▼                                            │
│            run_ingestion.py  (DATA_SOURCE switch)              │
│                    │                                            │
│            ner_extractor.py  (spaCy NER)                       │
└─────────────────────────────────────────────────────────────────┘
```

### Node and Edge Schema — Neo4j

```
(:Paper)  ──[:CITES]──────────▶  (:Paper)
(:Paper)  ──[:AUTHORED_BY]────▶  (:Author)
(:Paper)  ──[:RELATED_TO]─────▶  (:Concept)

Paper properties  : id, title, year, category, abstract,
                    paperId, corpusId, doi, venue, source,
                    citationCount, referenceCount
Author properties : key, name, authorId
Concept properties: name
```

### CITES Edge Modes

```
DATA_SOURCE=s2 + USE_REAL_CITATIONS=true
    └── Real reference-based CITES  (r.simulated = false)

DATA_SOURCE=arxiv  OR  no references in dataset
    └── Fallback concept-overlap CITES  (r.simulated = true)
        └── Two papers linked if they share ≥ CITES_THRESHOLD concepts
```

---

## 5. Data Flow

```
DATA_SOURCE env var
        │
        ▼
run_ingestion.py
        │
        ├── arxiv_fetcher.py ──────────────▶ data/arxiv_raw.json
        │                                   data/arxiv_clean.json
        │
        └── semantic_scholar_fetcher.py ──▶ data/s2_raw.json
                                            data/s2_clean.json
        │
        ▼
ner_extractor.py ─────────────────────────▶ data/<source>_ner.json
        │
        ├── chroma_store.py ──────────────▶ data/chromadb/
        │   (SPECTER encode + index)         (persistent vectors)
        │
        └── neo4j_store.py ───────────────▶ Neo4j AuraDB
            (Paper + Author + Concept         Paper nodes
             + CITES edges)                   CITES edges
        │
        ▼
retrieval.py
    neural_retrieve()  ◀── ChromaDB cosine search
    symbolic_expand()  ◀── Neo4j 1-2 hop CITES traversal
    nesy_retrieve()    ◀── merged + re-ranked result
        │
        ▼
validator.py
    validate_citations()  ◀── Neo4j direct node lookup
    (drops any ID not found in graph)
        │
        ▼
review.py / contradiction.py / hypothesis.py
    (LLM synthesis via Groq — Llama 3.3 70B)
        │
        ▼
metrics.py
    compute_all_metrics()  ◀── TS · NBR · ATD · RDI
        │
        ▼
orchestrator.py  (master entry point)
        │
        ▼
streamlit_app.py  (UI)
```

---

## 6. The Four-Stage Pipeline

### Stage 1 — Neural Retrieval
The user query is encoded into a 768-dimensional vector using SPECTER 2.0. ChromaDB performs cosine similarity search and returns the top-k most semantically similar papers.

### Stage 2 — Symbolic Expansion
The IDs of the top-k papers are sent to Neo4j. The graph is traversed 1–2 hops along CITES edges to surface highly-connected related papers that semantic search may have missed — papers that are conceptually adjacent but use different terminology.

### Stage 3 — Citation Validation
Every paper ID from stages 1 and 2 is verified against Neo4j with a direct node lookup. Any ID not corresponding to a real node is dropped before the LLM ever sees it. This is the **hallucination firewall** — architectural, not prompt-level.

### Stage 4 — LLM Synthesis
Verified papers are formatted into pipe-separated TOON context (Title|Year|Category|Abstract) and passed to Llama 3.3 70B via Groq. The LLM synthesises, analyses contradictions, or generates hypotheses depending on the mode.

---

## 7. Tech Stack

| Component | Tool | Why Chosen |
|-----------|------|-----------|
| Graph Database | Neo4j AuraDB Free | Native graph scaling, GDS library, Link Prediction |
| Embedding Model | SPECTER 2.0 (allenai-specter) | Pre-trained on scientific papers, 768-dim vectors |
| Vector Database | ChromaDB | Zero-latency local prototyping, persistent client |
| LLM | Llama 3.3 70B via Groq | Open weights, fast Groq inference |
| NLP / NER | spaCy en_core_web_sm | Fast, CPU-friendly noun chunk extraction |
| Frontend | Streamlit | Pure Python, rapid prototyping |
| Orchestration | LlamaIndex (planned) | Native Property Graph support |
| Data Source A | ArXiv API | 10k CS papers across 40 categories |
| Data Source B | Semantic Scholar Graph API | Real citation edges, bulk retrieval |

---

## 8. Codebase Structure

```
nesy-graphrag/
│
├── .env                              # Credentials — never commit
├── .env.example                      # Safe template
├── .gitignore
├── requirements.txt
├── README.md
│
├── data/                             # Gitignored — generated by pipeline
│   ├── arxiv_raw.json
│   ├── arxiv_clean.json
│   ├── arxiv_ner.json
│   ├── s2_raw.json
│   ├── s2_clean.json
│   ├── s2_ner.json
│   └── chromadb/
│
├── src/
│   ├── utils/
│   │   └── config.py                 # Single source of truth for all settings
│   │
│   ├── ingestion/
│   │   ├── arxiv_fetcher.py          # ArXiv API fetch + clean
│   │   ├── semantic_scholar_fetcher.py  # S2 bulk fetch + reference enrichment
│   │   ├── run_ingestion.py          # DATA_SOURCE dispatcher
│   │   └── ner_extractor.py          # spaCy NER + noise filter
│   │
│   ├── storage/
│   │   ├── chroma_store.py           # SPECTER encode + ChromaDB index + query
│   │   └── neo4j_store.py            # Neo4j insert + real/fallback CITES
│   │
│   └── pipeline/
│       ├── retrieval.py              # neural + symbolic + nesy_retrieve
│       ├── validator.py              # Citation existence firewall
│       ├── review.py                 # Literature review mode
│       ├── contradiction.py          # Contradiction detection mode
│       ├── hypothesis.py             # Hypothesis generation mode
│       ├── metrics.py                # TS · NBR · ATD · RDI  ← NEW
│       └── orchestrator.py           # Master entry point
│
└── app/
    └── streamlit_app.py              # Full Streamlit UI
```

---

## 9. What Has Been Done

### Phase 1 — Semester 5 (Aug–Dec 2025) ✅

- Defined the problem: information overload, LLM hallucination, passive tools
- Conducted 10-paper literature survey establishing academic foundation
- Identified 4 research gaps that existing systems do not address
- Proposed the NeSy-GraphRAG architecture
- Selected datasets (ArXiv, Hugging Face Scientific Papers, S2ORC)
- Defined evaluation metrics (TS, RDI, HNS)
- Submitted Phase 1 dissertation report

### Phase 2 — Semester 6 ✅

**Data Ingestion**
- Fetched 10,000 CS papers from ArXiv API across all 40 CS categories, 2,000 per year for 2020–2024
- Applied text cleaning: removed LaTeX math, commands, URLs, special characters, lowercased
- Dropped papers with missing titles/abstracts or abstracts under 30 words
- Built Semantic Scholar fetcher using `/paper/search/bulk` + `/paper/batch` for real citation references
- Implemented `DATA_SOURCE` switch in `config.py` and `run_ingestion.py` dispatcher

**Semantic Indexing**
- Loaded SPECTER 2.0 via sentence-transformers
- Encoded all 9,990 abstracts into 768-dim vectors in batches of 64
- Stored in ChromaDB with cosine similarity metric
- Resume support — already-stored papers skipped on re-run
- Source-aware collection naming (`arxiv_papers` vs `s2_papers`)

**Named Entity Recognition**
- spaCy `en_core_web_sm` on each abstract (truncated to 1,000 chars)
- Extracted ORG, PRODUCT, GPE, WORK_OF_ART, EVENT entities + noun chunks ≤4 words
- Rule-based noise filter: removed stopwords, single words under 4 chars, article-prefixed phrases

**Knowledge Graph (Neo4j)**
- Three node types: Paper, Author, Concept
- Two relationship types: AUTHORED_BY, RELATED_TO
- Batch insertion of 500 using UNWIND Cypher
- Real CITES from S2 references when available
- Fallback concept-overlap CITES (shared ≥2 concepts) for ArXiv mode
- Result: ~9,990 Paper nodes, ~24,176 CITES edges

**NeSy Retrieval Pipeline**
- `neural_retrieve()` — SPECTER encodes query, ChromaDB returns top-k
- `symbolic_expand()` — Neo4j traverses 1–2 CITES hops
- `nesy_retrieve()` — merges both, boosts papers appearing in both, returns re-ranked top-k

**Citation Validator**
- `validate_citations()` — Neo4j MATCH on paper IDs, drops any not found
- Prints verified vs. blocked count on every query

**Contradiction Detection**
- Graph query finds pairs sharing ≥2 concepts from different years
- LLM prompt returns VERDICT + REASON + CLAIM 1 + CLAIM 2 per pair

**Hypothesis Generation**
- Neo4j query finds structural holes — papers sharing ≥2 concepts with no existing CITES edge
- LLM generates HYPOTHESIS + RATIONALE + POTENTIAL IMPACT per hole

**Streamlit UI**
- Sidebar mode selector and top-k slider
- Source badges: neural+symbolic (green), neural only (blue), symbolic only (orange)
- Contradiction verdict colour coding: red / yellow / blue
- Live graph stats in sidebar (paper count, CITES edge count)

**Repository**
- Refactored from monolithic notebook to modular Python package
- All credentials in `.env`, never hardcoded
- All modules independently runnable via `python -m src.<module>`
- `.env.example` with all required variables

### Phase 3 — In Progress (Semester 7) 🔄

**Evaluation Metrics Module — `src/pipeline/metrics.py` ✅**

All four metrics implemented and integrated into orchestrator and Streamlit:

- **TS** (Trustworthiness Score) — live, returning real values
- **NBR** (NeSy Boost Ratio) — live, returning real values
- **ATD** (Answer Temporal Diversity) — live, returning real values
- **RDI** (Reasoning Depth Index) — live, known string-matching bug documented below

**Orchestrator integration** — `graphrag_query(mode="review")` now automatically calls `compute_all_metrics()` and stores scores in the result dict under `result["metrics"]`.

**End-to-end validation — April 22, 2026 ✅**

Full pipeline ran successfully on query `"graph neural networks for node classification"`:

```
TEST 1 — LITERATURE REVIEW
  Retrieved     : 10/10 papers
  Verified      : 10/10 citations in Neo4j
  LLM answer    : 3-paragraph synthesis with bracketed citations ✅
  TS            : 1.0000  ✅
  NBR           : 1.0000  ✅
  ATD           : 0.8000  (4/5 years — 2024 not retrieved for this query) ✅
  RDI           : 0.0000  (expected — consensus research area) ✅

TEST 2 — CONTRADICTION DETECTION
  Candidate pairs found  : 5
  LLM verdicts returned  : 5 × AGREEMENT
  Reason                 : GNN expressivity papers 2020–2023 are cumulative,
                           not contradictory — correct result for this query ✅

TEST 3 — HYPOTHESIS GENERATION
  Structural holes found : 10
  Hypotheses generated   : 10
  Notable connections    : adversarial robustness, imbalanced graphs,
                           hypergraph transformers, fault diagnosis GNNs,
                           distribution shift robustness ✅

COMBINED METRICS (r1 + r2)
  RDI : 0.0667  ← 1 false positive from string matching bug (documented)
```

---

## 10. Evaluation Metrics — Implemented

All four metrics live in `src/pipeline/metrics.py`.

### TS — Trustworthiness Score

```
TS = 0.5 × Citation_Integrity + 0.5 × (1 − Hallucination_Rate)

Citation_Integrity  = |Verified IDs in Neo4j| / |Total IDs retrieved|
Hallucination_Rate  = |Titles in answer NOT in verified set| / |Titles in answer|
```

**Range:** [0, 1]. Target ≥ 0.90. Current result: **1.0** ✅
**Status:** Reliable. Uses `validated_citations()` output directly.

---

### NBR — NeSy Boost Ratio

```
NBR = |Papers from graph expansion| / |Total retrieved|
```

Papers tagged `source = "symbolic"` or `"both"` count as graph-expanded.
**Range:** [0, 1]. Target > 0.30. Current result: **1.0**
**Status:** ⚠️ See known bugs — `source` field reliability under investigation.

---

### ATD — Answer Temporal Diversity

```
ATD = |Distinct years in retrieved papers| / span_size
```

`span_size` = 5 for the 2020–2024 dataset.
**Range:** [0, 1]. Target ≥ 0.60. Current result: **0.80** ✅
**Status:** Reliable. Pure computation on `year` field, no external dependencies.

---

### RDI — Reasoning Depth Index

```
RDI = (Sources_Used + Contradictions_Detected) / (Total_Sources + Total_Contradictions)

Sources_Used            = papers with source == "both"
Contradictions_Detected = pairs where LLM verdict == CONTRADICTION
```

**Range:** [0, 1]. Target ≥ 0.75. Current result: **0.0 / 0.0667**
**Status:** ⚠️ Two known issues — `source` field bug affects cross-doc component, string matching false positive affects contradiction component. See known bugs.

---

### HNS — Hypothesis Novelty Score (planned)

```
HNS = Average shortestPath(Concept_A, Concept_B) in Neo4j
```

**Status:** ⬜ Not yet implemented. Requires Neo4j `shortestPath` query across concept nodes. Deferred to Phase 3 completion.

---

## 11. Live Pipeline Output Analysis

This section documents what the April 22, 2026 orchestrator run actually means.

### What the Literature Review output shows

The LLM wrote a 3-paragraph synthesis correctly identifying the real trend in GNN research — the field moving away from message-passing frameworks toward more expressive architectures. Paper citations in brackets (`[benchmarking graph neural networks]`, `[ordered subgraph aggregation networks]`) all correspond to real papers in the graph. Zero fabricated citations. The one-line field summary is academically toned and accurate.

### What the Contradiction Detection output shows

Five candidate pairs were found. All five returned AGREEMENT. This is **correct** for this query. GNN expressivity papers from 2020–2023 form a cumulative research thread — each paper acknowledges and extends the previous one's limitations. To get meaningful contradiction results, queries on genuinely contested topics are needed (e.g., "transformer vs LSTM for sequence modelling", "batch normalisation effectiveness").

### What the Hypothesis Generation output shows

Ten structural holes were found. The 10 generated hypotheses bridge node classification with: adversarial robustness, imbalanced graph learning, hypergraph transformers, industrial fault diagnosis, and distribution shift robustness. These are legitimate under-explored connections. Hypothesis 9 (connecting gearbox fault diagnosis GNNs to node classification) is the type of cross-domain bridge that would score high on HNS.

### What the metric values mean in plain terms

| Metric | Value | Plain Meaning |
|--------|-------|--------------|
| TS = 1.0 | Perfect | No hallucinated citations reached the LLM |
| NBR = 1.0 | Suspicious | Every paper tagged as graph-sourced — source field needs investigation |
| ATD = 0.8 | Good | System drew from 4 of 5 years — healthy temporal spread |
| RDI = 0.0 | Expected | No contradictions in a consensus research area — correct behaviour |

---

## 12. Setup and Installation

### Prerequisites

- Python 3.10+
- Neo4j AuraDB Free account: https://neo4j.com/cloud/platform/aura-graph-database/
- Groq API key: https://console.groq.com
- Semantic Scholar API key (for S2 mode): https://www.semanticscholar.org/product/api

### Steps

```bash
# 1. Clone
git clone <your-repo-url>
cd nesy-graphrag

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Credentials
cp .env.example .env
# Edit .env — minimum required fields:
# DATA_SOURCE, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
# GROQ_API_KEY, SEMANTIC_SCHOLAR_API_KEY

# 5. Data directory
mkdir -p data/chromadb
```

---

## 13. Running the Pipeline

Run these in order. Each step produces a file the next step reads.

```bash
# Step 1 — Ingest papers
python -m src.ingestion.run_ingestion

# Step 2 — NER extraction (~10–15 min)
python -m src.ingestion.ner_extractor

# Step 3 — Build ChromaDB vector index (~20–30 min)
python -m src.storage.chroma_store

# Step 4 — Populate Neo4j (~5–10 min)
python -m src.storage.neo4j_store

# Step 5 — Full pipeline smoke test
python -m src.pipeline.orchestrator

# Step 6 — Streamlit UI
streamlit run app/streamlit_app.py
```

### Quick test with small dataset (no full ingestion)

```bash
DATA_SOURCE=s2 S2_LIMIT=20 S2_PAGE_SIZE=20 S2_BATCH_SIZE=20 \
python -m src.ingestion.run_ingestion
```

---

## 14. What Is Pending

### Phase 3 — Remaining

| Task | Priority | Notes |
|------|----------|-------|
| Fix `source` field in `nesy_retrieve()` | 🔴 High | NBR and RDI both depend on this being accurate |
| Fix RDI string matching — use `"VERDICT: CONTRADICTION"` not `"CONTRADICTION"` anywhere | 🔴 High | Currently produces false positives |
| Implement HNS via Neo4j `shortestPath` | 🟡 Medium | Requires live Neo4j, deferred until S2 data loaded |
| Baseline comparison — vector-only RAG vs NeSy-GraphRAG | 🔴 High | Core dissertation result — quantitative TS/NBR difference |
| Ground-truth benchmark queries with expected outputs | 🔴 High | Needed to validate TS and RDI against known answers |
| PDF ingestion pipeline | 🟡 Medium | S2ORC-style extraction, multi-column layout handling |
| Scale to 100k+ papers via S2ORC | 🟡 Medium | Requires Pinecone migration first |
| ChromaDB → Pinecone migration | 🟡 Medium | Required for concurrent users and larger datasets |
| PyVis citation graph in Streamlit UI | 🟢 Low | Interactive graph exploration |
| Unit tests for all modules | 🟡 Medium | Cover empty queries, Neo4j failures, fabricated IDs |
| NLI model for contradiction detection | 🟡 Medium | Replace structural proxy with fine-tuned BERT |
| Hypothesis feasibility check | 🟢 Low | Cross-reference generated hypotheses against literature |

### Phase 4 — Not Started

| Task | Notes |
|------|-------|
| Quantitative results writeup | Requires Phase 3 evaluation to be complete |
| Academic paper submission | Target: EMNLP / ACL / ICLR workshops on Scientific NLP |
| Patent application for citation-validation-as-hallucination-firewall | Pending IP review |
| Open-source release | After IP review |

---

## 15. Known Bugs and Limitations

### Bug 1 — NBR source field (medium severity)

**What:** `nesy_retrieve()` in `retrieval.py` tags the `source` field on papers but the tagging logic may be overwriting `"neural"` papers as `"both"` or `"symbolic"` incorrectly. NBR = 1.0 on every query is not credible.

**Effect:** NBR is inflated. RDI's `cross_doc` component reads the same field and shows 0 despite NBR showing 10/10 — the contradiction confirms the field is unreliable.

**Not yet fixed** — documented here for Phase 3 resolution.

---

### Bug 2 — RDI string matching false positive (low severity)

**What:** `compute_rdi()` checks `"CONTRADICTION" in analysis.upper()`. If the LLM says "this does NOT constitute a CONTRADICTION", it gets counted as resolved.

**Effect:** RDI gets a small false positive boost. In the April 22 run, 1 of 5 AGREEMENT verdicts was miscounted, producing RDI = 0.0667 instead of 0.0.

**Fix:** Change the check to `"VERDICT: CONTRADICTION" in analysis.upper()`.

**Not yet fixed** — documented here.

---

### Limitation 1 — CITES edge quality in ArXiv mode

ArXiv data has no real citation references. The fallback concept-overlap CITES edges are a proxy — two papers are linked if they share ≥2 NER-extracted concepts. This means the symbolic expansion in Stage 2 is traversing inferred relationships, not real citations. S2 mode with `USE_REAL_CITATIONS=true` resolves this.

---

### Limitation 2 — NER quality

spaCy `en_core_web_sm` is a general-purpose small model. It struggles with highly technical academic terms like "Weisfeiler-Leman", "GNN", "BERT". Concepts extracted are often generic noun chunks rather than precise technical entities. This directly affects the quality of concept-overlap CITES edges and the hypothesis structural holes.

---

### Limitation 3 — Contradiction detection on consensus topics

The graph-based candidate selection (shared concepts + different years) is a weak proxy for genuine scientific contradiction. It surfaces papers that discuss the same topic across time, not papers that genuinely disagree. On consensus research threads like GNN expressivity, all pairs will return AGREEMENT. This is correct behaviour but limits the system's usefulness on non-contested queries.

---

### Limitation 4 — 10,000 paper ceiling

At this scale the graph is sparse. Hypothesis generation finds structural holes, but many of them are coincidental overlaps rather than meaningful cross-domain bridges. HNS scoring will be more meaningful at 100k+ papers where the graph has enough density to distinguish near-connections from truly novel bridges.

---

### Limitation 5 — Single-user ChromaDB

The local persistent ChromaDB client does not support concurrent users. Any public deployment requires migration to Pinecone or a hosted vector database.

---

## 16. What Can Be Improved

### Short term (before dissertation submission)

**Contradiction detection** should use a proper NLI model rather than relying on the LLM returning a specific verdict string. A fine-tuned BERT on a contradiction dataset (following the DisContNet approach) would give deterministic, reproducible results that can be evaluated with CDP (Contradiction Detection Precision).

**The RDI formula** needs rethinking for the review-only case. Currently RDI = 0 if no contradiction mode has been run, which makes it uninformative as a standalone review metric. An alternative cross-document component — such as counting how many unique categories the retrieved papers span — would make RDI meaningful in all three modes.

**ATD should flag 2024 absence explicitly** in the Streamlit UI, not just report the number. If the most recent year is always absent from results, that suggests the embedding model or the query is biased toward older, more-cited papers.

### Medium term (Phase 3 completion)

**The baseline comparison** is the most important missing piece. Running the same queries through ChromaDB-only retrieval (skipping Neo4j entirely) and measuring TS, ATD, and answer quality would give the quantitative proof that the NeSy architecture actually outperforms standard RAG. Without this comparison, the dissertation evaluation chapter has no grounding.

**S2 data with real CITES edges** would transform the system. Real citation edges mean symbolic expansion actually follows how researchers cite each other, not just which papers happen to mention the same noun chunks. The `semantic_scholar_fetcher.py` is already implemented — this is a credentials and re-ingestion task, not an engineering task.

**HNS implementation** would complete the metrics suite. The `shortestPath` Cypher query is straightforward once Neo4j is loaded with S2 data. It would let you rank hypotheses by novelty and filter out low-value connections automatically.

### Long term (Phase 4)

**Domain-specific NER** trained on scientific text (e.g., SciBERT-based NER) would significantly improve concept extraction quality and therefore the quality of both the CITES fallback edges and the hypothesis structural holes.

**Multi-hop reasoning** beyond 2 hops is currently disabled for performance reasons. With Pinecone + a larger Neo4j instance, 3-hop traversal would surface genuinely non-obvious connections that 2-hop misses.

**User-uploaded PDFs** would let researchers add their own unpublished work to the index and ask the system to position it relative to the existing literature — a genuinely useful research tool beyond the current prototype scope.

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

---

*Last updated: April 22, 2026 — Post Phase 2 completion, Phase 3 in progress*
*Next update due: After baseline comparison results and S2 data ingestion*