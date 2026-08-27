# NeSy-GraphRAG — Pending Changes & Improvement Plan

**Status as of:** Phase 2 Complete / Phase 3 In Progress
**Purpose:** Track every known bug/gap and the concrete change needed to fix it, organized so three workstreams can proceed in parallel without blocking each other.

---

## 0. Parallelization Setup (do this first, once)

Before splitting into sections, generate frozen fixture files so Section 2 and Section 3 can work without live Neo4j/ChromaDB/Groq access.

- [ ] Run `orchestrator.py` once against the current (even buggy) pipeline for 2–3 test queries
- [ ] Save output to:
  - `fixtures/papers_sample.json` — output of `nesy_retrieve()`
  - `fixtures/review_result_sample.json` — output of `llm_review()`
  - `fixtures/contradiction_result_sample.json` — output of `llm_contradict()`
- [ ] Document the frozen interface contract (see bottom of this file) so nobody breaks the shape without a version bump
- [ ] Set a weekly integration checkpoint: run `orchestrator.py` live end-to-end, confirm sections still snap together, refresh fixtures if needed

**Rule:** Section 1 owns refreshing the fixtures. Sections 2 and 3 default to developing against fixtures, not the live stack.

---

## SECTION 1 — Ingestion → Storage → Retrieval

Files: `arxiv_fetcher.py`, `semantic_scholar_fetcher.py`, `run_ingestion.py`, `ner_extractor.py`, `chroma_store.py`, `neo4j_store.py`, `retrieval.py`

### 🔴 High priority

- [ ] **Fix `source` field bug in `nesy_retrieve()` (`retrieval.py`)**
  Merge logic in `nesy_retrieve()` is mislabeling papers as `"both"`/`"symbolic"` incorrectly, causing NBR to always read 1.0. Trace through the `seen[p["id"]]["source"] = "both"` assignment and confirm it only fires when a paper genuinely appears in both `neural_papers` and `symbolic_papers`.

- [ ] **Run full Semantic Scholar ingestion at scale**
  `semantic_scholar_fetcher.py` is fully implemented but only smoke-tested on 19 papers. Run it at target scale (`S2_LIMIT`, `S2_PAGE_SIZE` set to production values) to get real `CITES` edges instead of the arXiv concept-overlap proxy.
  - Requires: rotated `SEMANTIC_SCHOLAR_API_KEY`, `NEO4J_URI/USERNAME/PASSWORD`, `GROQ_API_KEY` in `.env`

- [ ] **Re-tune `CITES_THRESHOLD` and `HOP_DEPTH`** in `config.py` once real S2 CITES edges exist — current values were tuned against the synthetic concept-overlap graph and may not transfer.

### 🟡 Medium priority

- [ ] **Upgrade NER beyond `en_core_web_sm`** (`ner_extractor.py`)
  Swap in scispaCy or a SciBERT-based extractor. Current model produces generic noun-chunks and misses technical terms (e.g., "Weisfeiler-Leman"). This directly weakens CITES fallback edges and hypothesis-generation quality.

- [ ] **PDF ingestion pipeline** — S2ORC-style extraction for multi-column layouts, LaTeX, tables. New capability, not a fix.

### 🟢 Low priority

- [ ] ChromaDB → Pinecone migration (needed only for concurrent users / >100k papers)
- [ ] Scale to full S2ORC corpus (100k+ papers)
- [ ] Multi-hop (3+) traversal experiments in `symbolic_expand()`

### Frozen output contract (do not break without notifying Sections 2 & 3)
```
nesy_retrieve(driver, query, top_k) -> list[dict]
{
  "id": str, "title": str, "abstract": str,
  "year": int, "category": str, "score": float,
  "source": "neural" | "symbolic" | "both"
}
```

---

## SECTION 2 — Validation → LLM Synthesis

Files: `validator.py`, `review.py`, `contradiction.py`, `hypothesis.py`

**Can be developed entirely against `fixtures/papers_sample.json` — no live Neo4j/ChromaDB needed. Requires only a Groq key (or a mocked Groq client for prompt-only work).**

### 🔴 High priority

- [ ] **Deduplicate prompt templates across 4 files**
  The literature-review prompt is currently copy-pasted in `review.py` AND `streamlit_app.py` (already drifting risk). Contradiction and hypothesis prompts are similarly embedded inline. Extract all three into a shared `src/pipeline/prompts.py` module with functions like `build_review_prompt(toon, query)`.

- [ ] **Add error handling to `llm_review()`** (`review.py`)
  `contradiction.py` and `hypothesis.py` already wrap Groq calls in try/except with a fallback string; `review.py` does not. Bring it in line.

### 🟡 Medium priority

- [ ] **Replace contradiction-detection heuristic with a real NLI/BERT model**
  Current approach (shared ≥2 concepts + different years → ask LLM to verdict) always returns AGREEMENT on consensus topics — it's a weak proxy, not real contradiction detection. Implement a DisContNet-inspired fine-tuned BERT classifier as originally scoped in Phase 1. This only touches `contradiction.py` internals — output shape (`{query, contradictions}`) stays the same.

- [ ] **Add retry/backoff on Groq calls** — reuse the rate-limit/backoff pattern already implemented in `semantic_scholar_fetcher.py`'s `SemanticScholarClient` rather than writing new logic.

### 🟢 Low priority

- [ ] Fallback LLM model config (in case `LLM_MODEL` gets deprecated again, as happened once already per README)
- [ ] Hypothesis feasibility check — cross-reference generated hypotheses against existing literature before surfacing them

### Frozen output contract (do not break without notifying Section 3)
```
llm_review()      -> {"query": str, "papers": list, "answer": str, "verified": dict}
llm_contradict()  -> {"query": str, "contradictions": list[dict]}
llm_hypothesis()  -> {"query": str, "hypotheses": list[dict]}
```

---

## SECTION 3 — Metrics + UI

Files: `metrics.py`, `streamlit_app.py`, `orchestrator.py`

**Can be developed entirely against `fixtures/review_result_sample.json` and `fixtures/contradiction_result_sample.json` — no Groq/Neo4j/ChromaDB needed.**

### 🔴 High priority

- [ ] **Fix RDI string-matching false positive** (`metrics.py`, `compute_rdi()`)
  Change `"CONTRADICTION" in analysis.upper()` → `"VERDICT: CONTRADICTION" in analysis.upper()`. Currently miscounts verdicts like "this does NOT constitute a CONTRADICTION" as resolved. One-line fix, fully self-contained.

- [ ] **Build the baseline-comparison harness**
  Add a `vector_only_review()` path (or a flag on `llm_review()`) that skips `symbolic_expand()` and runs ChromaDB-only retrieval. Run the same query set through both paths, capture TS/NBR/ATD/RDI for each, and diff them. This is the core quantitative result the dissertation currently lacks — flagged repeatedly as the single biggest missing piece.

### 🟡 Medium priority

- [ ] **Refactor `streamlit_app.py` to call pipeline functions instead of reimplementing prompts inline**
  The review-mode prompt is currently duplicated in the UI file. Once Section 2 ships the shared `prompts.py` (or the functions themselves), swap the UI's inline prompt-building for direct calls to `llm_review()`/`llm_contradict()`/`llm_hypothesis()`. Pure UI cleanup — depends on Section 2's dedup work but not on live data.

- [ ] **Implement HNS (Hypothesis Novelty Score)**
  Planned since Phase 1, never built. Requires Neo4j GDS `shortestPath` between concept nodes on the two ends of a generated hypothesis. Additive — does not touch TS/NBR/ATD/RDI.

- [ ] **Add a results-logging layer**
  Append each run's metrics (query, mode, TS/NBR/ATD/RDI/HNS, timestamp) to a CSV or JSON log so trends can be plotted across many queries — needed to support the evaluation chapter with more than single-run snapshots.

### 🟢 Low priority

- [ ] Flag missing years (e.g., 2024 absence) explicitly in the Streamlit UI rather than only reporting the ATD number
- [ ] PyVis interactive citation-graph visualization in the UI
- [ ] Unit tests: fabricated paper IDs (confirm the firewall blocks them), empty query results, Neo4j connection failures

### Frozen input contract
This section only consumes `{papers, answer, verified}` (+ optional `contradictions`) — it never needs to know how those were produced.

---

## Suggested Execution Order (if working solo / sequentially instead of in parallel)

1. Section 3 → fix RDI bug (5 min, unblocks trustworthy metrics)
2. Section 1 → fix `source` field bug (unblocks trustworthy NBR)
3. Section 1 → run full S2 ingestion for real CITES edges
4. Section 3 → build baseline-comparison harness (now metrics are trustworthy and data is real)
5. Section 2 → dedupe prompts, add error handling
6. Section 3 → implement HNS
7. Section 1 → upgrade NER
8. Section 2 → replace contradiction heuristic with real NLI model
9. Everything else (tests, PDF ingestion, Pinecone migration, PyVis) — Phase 4 polish