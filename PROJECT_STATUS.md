# NeSy-GraphRAG Project Status

Last updated: August 31, 2026

Primary branch: `master`

Current milestone: abstract-based prototype complete; provisional retrieval evaluation complete

This file is the source of truth for implemented capabilities, evidence, limitations, and next priorities. PDF ingestion and full-text section extraction remain intentionally outside the current scope.

## Executive Summary

NeSy-GraphRAG is an end-to-end research prototype. It ingests scientific-paper metadata and abstracts, extracts concepts, builds Chroma and Neo4j stores, combines neural retrieval with citation-graph expansion, generates passage-cited literature reviews, evaluates contradiction candidates, generates evidence-ranked hypotheses, and exposes these workflows through Streamlit.

Since the core prototype milestone, the repository has gained a separate retrieval-evaluation framework with a frozen 20-query set, development/test splits, method-hidden candidate pooling, 1,329 provisional relevance judgments, standard information-retrieval metrics, vector/graph/hybrid comparisons, and paired significance analysis. Production retrieval now filters weak graph neighbours before fusion. The local corpus has also been expanded from the original graph-neural-network run to 11 CSE topic queries with full ID-preservation checks.

The implementation is working, but the research claims are not final. The current relevance judgments were produced by a fast machine-assisted title-and-abstract pass against the earlier corpus and require regeneration or extension plus human review for the expanded stores. The historical tuned hybrid result was not statistically better than vector-only retrieval.

## Verified Snapshot

### Data And Stores

- Semantic Scholar unique raw records: 52,822
- Records retained after cleaning: 47,619
- NER records processed: 47,619
- Configured topic queries: 11
- Clean papers attributed to multiple topic queries: 5,722
- Chroma collection: `s2_papers`
- Chroma vectors: 47,619
- Neo4j papers: 47,619
- Neo4j authors: 145,957
- Neo4j concepts: 195,252
- Neo4j `AUTHORED_BY` relationships: 190,128
- Neo4j `RELATED_TO` relationships: 475,121
- Neo4j real `CITES` relationships: 22,370
- Simulated `CITES` relationships in the current S2 graph: 0
- Papers participating in at least one citation edge: 13,624
- Papers with outgoing real citations: 5,947

All 10,000 original raw IDs and all 8,850 original clean/NER/store IDs were present after expansion. The current Python 3.11 rebuild resolves the live snapshot at 195,252 concepts; historical 43,581 and 42,937 counts belong to different 8,850-paper builds.

### Compute

- Automatic device selection: CUDA, then MPS, then CPU
- Primary local GPU verification: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM
- Primary CUDA build: PyTorch `2.12.1+cu126`
- SPECTER embedding inference verified on CUDA
- CPU-only evaluation also verified under Python 3.13.14
- NER supports configurable CPU multiprocessing and optional spaCy GPU use

### Tests And Live Checks

- 127 pytest cases pass under the current Python 3.11 environment
- `src`, `app`, and `tests` compile successfully
- `pip check` reports no broken requirements
- Chroma currently reports 47,619 vectors with exact clean-data ID parity
- Neo4j currently reports 47,619 papers with exact clean-data ID parity
- Eleven representative topic queries completed against live Chroma and Neo4j
- Latest recorded review smoke test: 5/5 claims accepted with valid passage references
- Latest recorded UI checks completed for review, contradiction, and hypothesis modes

## Retrieval Evaluation

### Benchmark

- Benchmark version: `0.2-draft`
- Status: `corpus_expanded_judgments_require_refresh_and_human_review`
- Frozen queries: 20
- Development queries: 6
- Held-out test queries: 14
- Provisional query-paper judgments: 1,329
- Relevance scale: 0 = not relevant, 1 = partially relevant, 2 = directly relevant
- Judgment source: fast title-and-abstract relevance pass
- Primary metric: NDCG@10
- Secondary metric: Recall@10

The query definitions are frozen, but the candidate pool and relevance labels predate the corpus expansion. Refresh the pool and judgments, then have humans review and correct them before reporting expanded-corpus results.

### Evaluation-Only Ablation

The evaluation package compares vector-only, citation-graph-only, and a two-way vector-plus-graph hybrid. The tuned evaluation hybrid uses a 16:1 vector-to-graph weight selected on the development split.

Historical held-out 14-query test results on the earlier 8,850-paper corpus:

| Method | NDCG@10 | Recall@10 | MAP | MRR |
|---|---:|---:|---:|---:|
| Evaluation hybrid | 0.3676 | 0.1440 | 0.1885 | 0.6500 |
| Vector only | 0.3643 | 0.1440 | 0.1886 | 0.6500 |
| Graph only | 0.2399 | 0.0841 | 0.1125 | 0.5133 |

Hybrid versus vector NDCG@10:

- Mean delta: `+0.0033` (approximately `+0.91%`)
- Bootstrap 95% interval: `[0.0000, 0.0099]`
- Bootstrap probability of a positive delta: `0.6456`
- Exact two-sided randomization p-value: `1.0`

This result does not establish a statistically significant hybrid advantage.

### Production Retrieval Filter

The application pipeline remains separate from the evaluation-only 16:1 hybrid. Production uses 1:1 reciprocal-rank fusion after filtering graph candidates by stored SPECTER similarity, meaningful query-term coverage, and distinct vector-seed connections.

Historical held-out comparison recorded after development-only threshold selection on the earlier corpus:

| Production flow | NDCG@10 | Recall@10 | Mean graph papers retained |
|---|---:|---:|---:|
| Old unfiltered hybrid | 0.2678 | 0.1117 | 20.00 |
| New filtered hybrid | 0.3617 | 0.1422 | 0.79 |
| Vector-only reference | 0.3643 | 0.1440 | 0.00 |

Filtering removes the large relevance loss caused by weak graph neighbours. It does not show that production GraphRAG beats vector-only retrieval. The graph currently contributes occasional strong discoveries rather than a consistent ranking improvement.

## Implemented

### Core Data Pipeline

- ArXiv and Semantic Scholar ingestion
- Resumable multi-topic S2 ingestion with completed-topic checkpoints
- Global paper-ID deduplication and merged query/reference provenance
- Existing raw-corpus preservation during topic expansion
- Incremental NER reuse and atomic checkpoints every 5,000 new papers
- Real Semantic Scholar reference IDs
- Cleaning, filtering, retries, batching, and JSON persistence
- spaCy entity and noun-chunk extraction
- Persistent SPECTER/Chroma indexing
- Neo4j papers, authors, concepts, `AUTHORED_BY`, `RELATED_TO`, and real `CITES`
- Resume support for Chroma indexing
- Placeholder-aware credential validation
- Explicit opt-in before destructive Neo4j rebuilds
- Automatic GPU/CPU resource selection

### Retrieval And Generation

- Vector retrieval with real cosine similarity
- Citation-graph expansion with seed self-match exclusion
- Weighted reciprocal-rank fusion and source labels
- Production graph-candidate relevance filtering
- Cached query embeddings and stored-paper embedding scoring
- Retrieval diagnostics for candidate filtering and ranking
- Neo4j paper validation before LLM context construction
- Deterministic sentence passage IDs
- Strict claim-to-passage provenance parsing
- Blocking and audit retention for malformed or fabricated passage references
- Structured literature-review rendering
- Groq retry and fallback behavior

### Contradictions And Hypotheses

- Cross-year contradiction candidate discovery
- Normalized concept-overlap ranking
- Strict `CONTRADICTION`, `AGREEMENT`, and `DIFFERENT SCOPE` parsing
- Confidence gating and malformed-output handling
- Structural-hole hypothesis candidates
- Evidence scores, supporting papers, and shared concepts
- Structured feasibility and missing-evidence fields
- Rejected hypothesis audit records

### Evaluation And Diagnostics

- Frozen 20-query dev/test retrieval set
- Method-hidden vector/graph/hybrid candidate pooling
- Strict benchmark and judgment validation
- CSV-to-judged-benchmark finalization
- Precision, Recall, Hit Rate, MRR, MAP, NDCG, unjudged rate, and latency
- Vector-only, graph-only, and two-way hybrid retrieval ablation
- Per-query rankings and metrics exports
- Bootstrap intervals and exact paired randomization testing
- Production graph-filter held-out comparison
- Prototype TS, NBR, ATD, RDI, and HNS diagnostics
- Versioned provenance-aware CSV logging

### Interface And Engineering

- Streamlit review, contradiction, and hypothesis modes
- Claim evidence and unsupported-output inspection
- Retrieval source labels and prototype metrics
- 127 deterministic pytest cases
- Pinned direct dependencies
- Windows setup instructions for local Neo4j, CUDA, and CPU
- Python 3.11 CUDA and Python 3.13 CPU verification records

## Not Yet Implemented Or Validated

- Human-finalized retrieval relevance judgments and a version `1.0` benchmark
- Independent or multi-reviewer agreement measurement
- BM25 lexical baseline in the current evaluation package
- Conventional non-graph RAG with the same generator and context budget
- Rule-based contradiction baseline
- Labeled scientific contradiction benchmark
- Contradiction precision, recall, F1, confusion matrix, and calibration evaluation
- Semantic entailment scoring for each review claim and cited passage
- Claim-support, citation precision/recall, and review-completeness benchmark
- Human-reviewed hypothesis samples and expert scoring rubric
- Scientific novelty and feasibility validation for hypotheses
- scispaCy or another scientific NER comparison
- SPECTER2 or other scientific embedding comparisons
- Scientific-domain versus general LLM comparison
- Full controlled ablations across provenance, validation, retrieval depth, and context budget
- Repeated stochastic model runs with uncertainty reporting
- Controlled scaling comparisons at 10K, 50K, 100K, and larger collections; one 47,619-paper operational build is complete but is not a scaling benchmark
- Indexing throughput, memory, GPU-memory, latency, and API-cost benchmark suite
- Evaluation comparison dashboard and exports in Streamlit
- Advanced graph exploration
- Continuous integration and live Neo4j/Chroma/Groq integration tests
- Platform-specific dependency lockfiles with hashes
- Production deployment, authentication, monitoring, and multi-user isolation
- Full-text/PDF ingestion and section-level provenance

## Limitations

### Evidence And Data

- The corpus is 47,619 records gathered through 11 CSE query strings, not a balanced CSE taxonomy or multidisciplinary million-scale collection.
- Most evidence is abstract text and metadata.
- Abstract evidence cannot reliably support detailed methods, tables, appendices, or limitations.
- A valid passage ID proves traceability, not semantic entailment or scientific correctness.

### Retrieval Evaluation

- Twenty queries are too few for broad generalization.
- The existing judged benchmark is concentrated in graph machine learning and predates the broad-CSE stores.
- The 1,329 labels are machine-assisted and require human review.
- The evaluation-only hybrid and production hybrid are different configurations.
- The tuned hybrid improvement over vector-only is not statistically significant.
- Graph-only retrieval is materially weaker than vector retrieval on the current benchmark.
- The graph remains sparse at 22,370 real citation edges for 47,619 papers, and generic noun chunks introduce noisy concept nodes.
- Development-selected thresholds need confirmation on a human-reviewed benchmark.

### Generated Analysis

- Contradiction judgments rely on candidate heuristics and a hosted LLM, not validated scientific NLI.
- LLM confidence values are not calibrated.
- Hypothesis feasibility is a generated judgment, not expert or experimental validation.
- Structural graph novelty is not equivalent to scientific novelty.
- Hosted model latency, behavior, availability, and cost remain external dependencies.

### Engineering

- Most tests use deterministic doubles; live services are not exercised in CI.
- Direct dependencies are pinned, but transitive dependencies lack hashed lockfiles.
- Historical graph snapshots used different corpora/Python environments; reproducible store and model manifests are still needed.
- Evaluation results are file-based rather than managed by a full experiment tracker.
- The UI is designed for the application workflow, not benchmark comparison.

## Priority Roadmap

### Priority 1: Finalize Retrieval Evidence

1. Regenerate the method-hidden candidate pool against the expanded corpus.
2. Record reviewer identity, instructions, disagreements, and adjudication.
3. Freeze benchmark version `1.0` without retuning on the test split.
4. Rerun vector, graph, evaluation hybrid, and production-filter evaluation.
5. Report the result even if vector-only remains best.

### Priority 2: Complete Retrieval Baselines

1. Add deterministic BM25.
2. Add conventional vector RAG with a matched generator and context budget.
3. Keep graph-only and hybrid ablations separate from application claims.
4. Compare latency and resource use as well as relevance.

### Priority 3: Evaluate Generated Outputs

1. Build or adopt a labeled scientific contradiction set.
2. Evaluate contradiction precision, recall, F1, and calibration.
3. Create a claim/passage entailment and citation-quality sample.
4. Create expert-reviewed hypothesis samples and rubrics.
5. Separate validated findings from model-generated suggestions in reports.

### Priority 4: Controlled Model Experiments

1. Compare spaCy with scientific NER alternatives.
2. Compare SPECTER with stronger scientific embeddings.
3. Compare general and scientific-domain LLMs.
4. Sweep fusion weights, retrieval depth, candidate limits, and `top_k` on development data only.
5. Run component ablations and repeated stochastic trials.

### Priority 5: Scaling, UI, And Reliability

1. Benchmark 10K, 50K, 100K, and larger corpora.
2. Record graph density, indexing time, latency, throughput, memory, and cost.
3. Add benchmark comparison and export views to Streamlit.
4. Add CI, marked live-service tests, structured logs, and experiment metadata.
5. Add reproducible store and model version manifests.

## Recommended Next Step

Refresh the candidate pool and relevance judgments against the expanded corpus, then human-review and freeze benchmark `1.0`. Add BM25 and rerun the held-out comparison before tuning retrieval logic. This gives every later NER, embedding, LLM, graph, and scaling experiment a trustworthy measuring instrument.

## Readiness

- Working abstract-based demo: approximately 95%
- Core capstone implementation: approximately 92%
- Retrieval-evaluation infrastructure: approximately 85%
- Defensible capstone evaluation: approximately 72%
- Research-grade system: approximately 40%

The implementation milestone is complete for the current scope. The next milestone is evidence validation, not more unmeasured feature work.
