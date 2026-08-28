# NeSy-GraphRAG Project Status

Last updated: August 29, 2026

Branch: `phase3`

Implementation milestone: claim-level prototype complete

This file is the single source of truth for current capabilities, missing work, limitations, and future priorities. PDF ingestion and full-text section extraction are intentionally outside the current scope.

## Executive Summary

NeSy-GraphRAG is an end-to-end working research prototype. It ingests scientific-paper metadata and abstracts, extracts concepts, builds Chroma and Neo4j stores, retrieves through neural and symbolic paths, generates literature reviews with sentence-level provenance, detects contradiction candidates, generates evidence-ranked hypotheses, and exposes the workflow through Streamlit.

The core implementation is sealed for the current scope. The project is not yet a validated research system. The remaining work is primarily experimental: build benchmarks, implement standard metrics and baselines, run controlled ablations, evaluate specialized scientific models, scale under measurement, and improve evaluation reporting.

## Verified Snapshot

### Data And Stores

- Semantic Scholar records fetched: 10,000
- Records retained after cleaning: 8,850
- NER records processed: 8,850
- Chroma collection: `s2_papers`
- Chroma vectors: 8,850
- Neo4j nodes:
  - `Paper`: 8,850
  - `Author`: 27,655
  - `Concept`: 43,581
- Neo4j relationships:
  - `AUTHORED_BY`: 37,180
  - `RELATED_TO`: 88,268
  - real `CITES`: 7,203
  - simulated `CITES`: 0
- Citation coverage:
  - 2,989 papers participate in at least one citation edge
  - 1,581 papers have outgoing real citation edges

### Compute

- Automatic device selection: CUDA, then MPS, then CPU
- Verified GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM
- Verified PyTorch build: `2.12.1+cu126`
- SPECTER retrieval selects CUDA successfully
- NER supports configurable CPU multiprocessing and optional spaCy GPU use

### Tests And Live Checks

- 33 pytest cases pass
- `src`, `app`, and `tests` compile successfully
- Pytest uses strict configuration and registered integration markers
- Latest claim-provenance smoke test:
  - 3/3 retrieved papers verified in Neo4j
  - 23 sentence passages created
  - 5/5 generated claims accepted
  - 9/9 passage citations valid
  - 0 unsupported claims
  - 0 parse errors
  - 1 generation attempt; repair was not needed

### Existing Retrieval Result

The corrected five-query comparison against vector-only retrieval produced these average deltas:

- TS: `0.00`
- NBR: `+0.56`
- ATD: `-0.12`
- RDI: `+0.10`

Per-query hybrid NBR was `0.5`, `0.5`, `0.5`, `0.7`, and `0.6`; vector-only NBR was `0.0`. This proves that graph results affect the final ranking. It does not prove improved relevance because the queries do not yet have relevance judgments.

## Implemented

### Ingestion And Cleaning

- ArXiv ingestion
- Semantic Scholar ingestion with real reference IDs
- Source selection through `DATA_SOURCE`
- Cleaning, filtering, batching, retries, and local JSON persistence
- Configurable limits, years, fields of study, publication types, and sorting

### Entity Extraction

- spaCy entity and noun-chunk extraction
- Batched processing
- Configurable multiprocessing through `SPACY_N_PROCESS`
- Configurable batch sizing through `SPACY_BATCH_SIZE`
- GPU preference with parallel CPU fallback

### Vector Retrieval

- Persistent Chroma storage
- SPECTER document and query embeddings
- Automatic CUDA/MPS/CPU selection
- Conservative GPU batching for 4 GB VRAM
- Resume support for partially indexed collections
- Cached/offline Hugging Face model loading
- CPU thread controls for PyTorch, OpenMP, and MKL

### Knowledge Graph

- Neo4j nodes for papers, authors, and concepts
- `AUTHORED_BY`, `RELATED_TO`, and `CITES` relationships
- Batched graph loading
- Real Semantic Scholar citation loading
- Simulated-citation fallback code for sources without references; unused in the current S2 graph
- Credential validation, connectivity checks, and safe driver cleanup

### Hybrid Retrieval

- Neural retrieval from Chroma
- Symbolic expansion through Neo4j citation traversal
- Real cosine similarity from Chroma
- Normalized graph-connectivity scoring
- Weighted reciprocal-rank fusion
- `neural`, `symbolic`, and `both` result labels
- Configurable fusion weights, `top_k`, hop depth, and RRF constant
- Vector-only retrieval path
- Retrieval diagnostics for ranks, scores, graph connections, citation degree, source distribution, and cutoff decisions

### Paper And Claim Provenance

- Retrieved paper IDs are checked against Neo4j before reaching the LLM
- Only verified-paper abstracts enter review context
- Abstracts are split into deterministic sentence passages
- Passage IDs use a stable paper-ID hash and sentence position
- Review prompts require every generated claim to cite supplied passage IDs
- Passage text is treated as untrusted source data in the prompt
- Missing, malformed, fabricated, and mixed-validity passage references are blocked
- Claims with incomplete citation sets are excluded from the displayed review
- Accepted claims, rejected claims, raw generations, parser errors, and passage metadata are retained
- Generation temperature is deterministic
- One bounded format-repair attempt runs only when the first response yields zero accepted claims
- Console rendering is safe under the default Windows encoding

Claim provenance guarantees traceability and passage-ID validity. It does not by itself prove that the cited sentence semantically entails the generated claim; that requires benchmark evaluation.

### Literature Review

- Hybrid or vector-only retrieval
- Verified passage context
- Structured claim/evidence generation
- Grounded-review rendering from accepted claims only
- Provenance statistics and unsupported-claim audit output
- Groq retries and model fallback

### Contradiction Detection

- Cross-year graph candidate generation
- Candidate ranking by normalized concept Jaccard and year gap
- Configurable overlap and shared-concept thresholds
- Structured verdicts: `CONTRADICTION`, `AGREEMENT`, and `DIFFERENT SCOPE`
- Exact parser and malformed-output handling
- Deterministic generation
- Confidence gating
- Shared verdict interpretation across pipeline, metrics, and UI
- Live checks have returned valid structured verdicts

### Hypothesis Generation

- Structural-hole candidate discovery
- Minimum query-paper support
- Evidence scores using concept overlap and query support
- Supporting paper IDs and shared concepts retained
- Structured hypothesis, feasibility, evidence, missing-evidence, rationale, and impact fields
- Only valid `HIGH` and `MEDIUM` feasibility generations accepted
- Rejected and malformed generations retained for audit

### Prototype Metrics And Logging

- TS: passage-citation integrity and accepted-claim coverage when provenance exists
- NBR: graph participation in final retrieval
- ATD: temporal diversity over a configured year range
- RDI: prototype cross-document and contradiction reasoning score
- HNS: prototype graph-path novelty score
- CSV evaluation logging
- Vector-only comparison harness

These are project diagnostics, not yet standard or independently validated research metrics.

### Streamlit UI

- Literature review, contradiction, and hypothesis modes
- Retrieval source badges and paper details
- Claim-to-sentence evidence inspection
- Unsupported-claim and parser audits
- Prototype metric display
- Basic Neo4j and Chroma statistics
- Hypothesis evidence, support, feasibility, and missing-evidence display

### Engineering Foundation

- Centralized configuration and prompt templates
- Shared Groq retry/fallback client
- Automatic compute-resource selection
- Native pytest tests, fixtures, parametrization, monkeypatching, and strict markers
- Unit guards for citation fabrication, metrics edge cases, retrieval fusion, Neo4j failures, contradiction parsing, hypothesis validation, provenance parsing, and repair behavior
- README and this status file are the maintained project documentation

## Not Implemented

The following are not part of the current implementation:

- A fixed, judged retrieval benchmark
- Standard IR evaluation: Precision@K, Recall@K, MRR, MAP, and NDCG
- Labeled scientific contradiction evaluation
- Human-reviewed hypothesis evaluation
- Frozen train/development/test splits
- BM25 baseline
- Standalone graph-only baseline harness
- Conventional non-graph RAG baseline with the same generator and context budget
- Rule-based contradiction baseline
- Repeated-run confidence intervals and statistical significance tests
- Semantic entailment verification between each claim and cited passage
- scispaCy or another scientific NER model
- SPECTER2 or systematic scientific-embedding comparisons
- Scientific-LLM comparison
- Local model training or fine-tuning
- Corpus scaling beyond the current 8,850 records
- Performance benchmarks at 10K, 50K, 100K, and larger scales
- Full-text or PDF ingestion
- Section-level provenance
- Advanced graph visualization
- Benchmark dashboards, comparison tables, and evaluation exports in the UI
- Continuous integration and automated live-service integration tests
- Production deployment, authentication, multi-user isolation, monitoring, or service-level objectives

## Limitations

### Data

- The corpus contains 8,850 mostly computer-science records, not the million-scale multidisciplinary corpus proposed in the Phase 1 report.
- Most evidence is abstract text and metadata rather than full paper content.
- Abstract-only evidence cannot support claims that depend on methods, tables, limitations, appendices, or detailed results.

### Graph

- The real citation graph is sparse: only 2,989 of 8,850 papers participate in citation edges.
- Scaling may improve coverage but will not automatically correct ranking or relevance logic.
- Generic noun chunks create noisy or overly broad concept nodes.
- One-hop or bounded citation expansion can favor well-connected papers regardless of query relevance.

### Retrieval

- NBR measures graph participation, not relevance improvement.
- The existing five-query comparison is too small and has no human relevance labels.
- Equal default fusion weights have not been selected through judged optimization.
- Hybrid retrieval reduced ATD by `0.12` on average in the existing comparison.

### Provenance And Generation

- A valid passage ID proves that the evidence exists and was supplied to the model; it does not prove semantic entailment.
- Structured-output repair improves format reliability but does not improve scientific correctness.
- Hosted LLM behavior, latency, availability, and cost remain external dependencies.
- Review generation is constrained to claims that can be represented in the strict parser format.

### Contradictions And Hypotheses

- Contradiction verdicts depend on candidate heuristics and a hosted LLM rather than a validated scientific NLI model.
- Confidence values are self-reported by the LLM and are not calibrated.
- Hypothesis feasibility is an LLM judgment, not experimental or expert validation.
- Structural holes indicate graph novelty, not necessarily scientific novelty or usefulness.

### Metrics

- TS is a prototype structural-grounding score, not a semantic factuality metric.
- NBR should be reported as a graph-contribution diagnostic only.
- ATD measures year coverage, not review quality.
- RDI is a custom formula and needs external justification or replacement.
- HNS currently uses `1 / path_length`, which rewards shorter paths while the documented novelty interpretation expects longer paths to indicate novelty. This direction mismatch must be corrected before final evaluation.

### Engineering And UI

- Unit tests use deterministic doubles; most external-service workflows are verified manually rather than in CI.
- Dependency versions are not fully locked for reproducible environments.
- The UI is functional but not yet designed for benchmark comparison or large evaluation runs.
- Evaluation CSV logging is basic and not a complete experiment-tracking system.

## Future Work

### Phase A: Evaluation Foundation

1. Define research questions and success criteria.
2. Freeze a representative query set.
3. Create paper-level relevance judgments.
4. Create or adopt labeled scientific contradiction pairs.
5. Create human-reviewed hypothesis samples and scoring rubrics.
6. Freeze train, development, and test partitions where applicable.
7. Record dataset versions, random seeds, prompts, models, and parameters.

### Phase B: Standard Metrics And Baselines

1. Add Precision@K, Recall@K, MAP, MRR, and NDCG.
2. Add contradiction precision, recall, F1, confusion matrix, and calibration measures.
3. Add claim-support, citation precision/recall, and review-completeness evaluation.
4. Add hypothesis evidence, novelty, feasibility, and expert-rating measures.
5. Add latency, throughput, GPU memory, indexing time, and API-cost measurements.
6. Implement BM25, graph-only, standard RAG, vector-only, and rule-based baselines.
7. Compare every baseline with the same corpus, queries, generator, and context budget.

### Phase C: Controlled Experiments

1. Sweep neural/graph fusion weights.
2. Compare graph hop depth and candidate-pool size.
3. Compare `top_k` and context budgets.
4. Run component ablations: vector only, graph only, fusion, paper validation, claim provenance, and complete NeSy pipeline.
5. Repeat stochastic experiments and report confidence intervals and significance tests.

### Phase D: Scientific Model Experiments

1. Compare current spaCy extraction with scispaCy or another scientific concept model.
2. Compare current SPECTER embeddings with stronger scientific retrieval embeddings.
3. Compare the current LLM with scientific-domain and general instruction models.
4. Change one model family at a time and measure downstream graph, retrieval, reasoning, latency, and cost effects.
5. Select models from benchmark results rather than domain branding alone.

### Phase E: Scaling

1. Improve citation and concept quality before increasing volume.
2. Benchmark at roughly 10K, 50K, 100K, and larger corpus sizes.
3. Measure indexing time, throughput, storage, memory, graph density, and retrieval latency.
4. Verify that quality remains stable as graph degree and candidate volume change.
5. Add incremental ingestion and reproducible store-version metadata.

### Phase F: UI And Reporting

1. Compare BM25, vector, graph, hybrid, and validated NeSy results side by side.
2. Display benchmark configuration, model versions, and experiment IDs.
3. Add retrieval source distributions and ranking explanations.
4. Add graph exploration for papers, concepts, citations, and evidence paths.
5. Export claims, citations, metrics, and experiment results.
6. Clearly distinguish validated evidence, rejected output, model interpretation, and generated hypotheses.

### Phase G: Engineering Reliability

1. Add marked Neo4j, Chroma, and Groq integration tests.
2. Add CI for tests, compilation, formatting, and static checks.
3. Lock critical dependency versions and document environment reproduction.
4. Replace basic CSV logging with versioned experiment records.
5. Add structured logging, failure telemetry, and resource measurements.

## Recommended Next Step

Start with Phase A and a small but carefully judged retrieval set. Then implement standard retrieval metrics and BM25 before changing NER, embeddings, the LLM, fusion settings, or corpus scale. This creates a fixed measuring instrument for every later decision.

## Readiness

- Working demo: approximately 95%
- Core capstone prototype implementation: approximately 90%
- Defensible capstone evaluation: approximately 62%
- Research-grade system: approximately 35%

The implementation milestone is complete for the current abstract-based scope. The research milestone will be complete only after benchmarked relevance, semantic support, reasoning quality, baseline comparisons, and controlled experiments are reported.
