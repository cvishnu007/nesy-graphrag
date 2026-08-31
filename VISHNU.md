# Vishnu - Core Prototype And Integration Work

Date range: August 27-31, 2026

Primary working branch: `phase3`

Merged destination: `master`

## Scope Completed

This work took NeSy-GraphRAG from a partially documented prototype to a verified abstract-based end-to-end system. The focus was real Semantic Scholar data, efficient local execution, hybrid retrieval, strict claim provenance, contradiction and hypothesis reliability, automated testing, reproducible setup, and repository cleanup.

Later retrieval-benchmark work by teammates is documented separately in `devesh.md` and `devesh2.md`. This file records the core implementation and integration work completed before and during that handoff.

## Repository And Documentation Cleanup

- Read the Phase 1 report and compared the proposed system with the codebase while intentionally excluding PDF ingestion from the current scope.
- Removed redundant phase/status files and obsolete change logs.
- Preserved the teammate work log now tracked as `Chinmay.md`.
- Established `PROJECT_STATUS.md` as the implementation and roadmap source of truth.
- Rewrote README content to match the actual code, models, corpus, and limitations.
- Added a detailed `SETUP.md` for Windows, local Neo4j, CUDA/CPU installation, environment configuration, full data rebuilds, tests, troubleshooting, and evaluation.
- Added the rule that contributors synchronize a clean `master` with `git pull --ff-only origin master` before beginning changes.
- Resolved merge conflicts between `master` and `phase3` without losing teammate history or the newer tested implementation.

## Data Pipeline And Stores

- Reused the existing virtual environment instead of reinstalling dependencies unnecessarily.
- Ran Semantic Scholar ingestion for 10,000 graph-neural-network papers from 2020-2025.
- Retained 8,850 cleaned papers after filtering.
- Completed NER for all 8,850 cleaned papers.
- Built the `s2_papers` Chroma collection with 8,850 vectors.
- Loaded 8,850 papers into Neo4j with authors, concepts, and real Semantic Scholar citation relationships.
- Verified 7,203 real `CITES` relationships and no simulated citations in the S2 graph.
- Added safer handling for empty datasets, missing columns, partial Chroma indexes, and invalid source configuration.
- Added explicit `NEO4J_ALLOW_RESET=true` protection before destructive graph rebuilds.
- Added placeholder-aware Neo4j, Groq, and Semantic Scholar credential checks.
- Added resumable broad-CSE ingestion across 11 configured topics.
- Added global paper-ID deduplication with merged topic and reference provenance.
- Preserved the existing raw corpus and checkpointed every completed topic atomically.
- Made NER reuse existing entity output and checkpoint every 5,000 new papers.
- Completed the 11-topic broad-CSE run with 52,822 unique raw records and 47,619 cleaned records.
- Verified zero missing IDs from the original 10,000 raw and 8,850 cleaned/NER/store records.
- Reused 8,850 NER records, processed 38,769 additions with 15 CPU workers, and verified 47,619 entity-bearing papers.
- Reused 8,850 Chroma vectors, encoded 38,769 additions on the RTX 3050, and verified exact 47,619-ID parity with the clean data.
- Rebuilt Neo4j with 47,619 papers, 145,957 authors, 195,252 concepts, and 22,370 real citation edges.
- Ran live retrieval smoke tests across all 11 configured CSE topics; vector retrieval broadened successfully, while strict graph filtering still retained few citation neighbours.

## Compute And Performance

- Added automatic CUDA, MPS, and CPU selection for embedding work.
- Verified CUDA inference on an NVIDIA GeForce RTX 3050 Laptop GPU with PyTorch `2.12.1+cu126`.
- Added configurable embedding batches, PyTorch CPU threads, spaCy workers, and spaCy device selection.
- Used all but one logical CPU by default for CPU-parallel stages while retaining environment overrides.
- Preserved conservative GPU batches for the 4 GB laptop GPU.
- Converted runtime console output that caused Windows encoding failures to portable ASCII output.

## Hybrid Retrieval

- Corrected neural and symbolic result fusion so graph-only papers can enter the final ranking.
- Added weighted reciprocal-rank fusion with configurable neural/graph weights and RRF constant.
- Preserved `neural`, `symbolic`, and `both` source labels.
- Added normalized graph-connectivity scores and real Chroma cosine similarity.
- Excluded graph seed self-matches and counted distinct vector-seed connections.
- Added retrieval diagnostics showing neural rank, graph rank, graph links, citation degree, final score, source distribution, and cutoff decisions.
- Added a vector-only retrieval path and comparison harness.

The later production graph-relevance filter and formal retrieval-evaluation package were added through subsequent teammate branches and are described in `devesh.md`, `devesh2.md`, and `PROJECT_STATUS.md`.

## Claim-Level Provenance

- Verified every retrieved paper ID against Neo4j before its abstract can reach the LLM.
- Split verified abstracts into deterministic sentence passages.
- Added stable passage IDs based on paper ID and sentence position.
- Required structured `CLAIM` and `EVIDENCE` output from the review model.
- Blocked claims containing missing, malformed, fabricated, or mixed-validity passage IDs.
- Displayed only claims with completely valid passage-reference sets.
- Retained unsupported claims, raw generations, parser errors, and passage metadata for audit.
- Added one bounded format-repair attempt when the first generation contains no accepted claim.
- Updated TS to use passage-citation integrity and accepted-claim coverage when provenance is available.
- Extended evaluation logging with versioned claim-provenance fields.

Claim provenance establishes traceability and passage existence. It does not prove semantic entailment, factual correctness, or scientific validity.

## Contradiction And Hypothesis Reliability

- Added normalized concept-Jaccard and year-gap scoring for contradiction candidates.
- Added strict parsing for `CONTRADICTION`, `AGREEMENT`, and `DIFFERENT SCOPE` verdicts.
- Added deterministic generation, confidence thresholds, and malformed-output handling.
- Unified verdict interpretation across pipeline code, metrics, tests, and Streamlit.
- Ranked structural-hole hypothesis candidates by concept overlap and query-paper support.
- Added structured feasibility, supporting evidence, missing evidence, rationale, and impact fields.
- Accepted only valid `HIGH` and `MEDIUM` feasibility outputs.
- Retained weak, malformed, and rejected hypothesis outputs for audit.

## Metrics And Interface

- Corrected prototype metric behavior for empty results and Neo4j failures.
- Corrected HNS so longer measured graph paths produce higher normalized structural novelty.
- Removed unsupported target labels that made prototype diagnostics look like validated benchmarks.
- Made ATD use the configured corpus year range.
- Updated the Streamlit controls so each mode uses its selected paper/candidate count.
- Added review claim-evidence inspection and unsupported-output audits to the UI.
- Clarified that contradiction pairs are evaluated rather than automatically verified.

## Testing And Reproducibility

- Added pytest as a pinned project dependency.
- Migrated the test suite to native pytest fixtures, parametrization, monkeypatching, and strict markers.
- Added tests for retrieval fusion, provenance parsing, fabricated citations, contradiction verdicts, hypothesis validation, metrics edge cases, Neo4j failures, reset protection, result logging, and configuration guards.
- Reached 39 passing tests at the core-prototype handoff.
- Pinned direct dependencies to the verified environment.
- Verified Python compilation and `pip check` before commits.
- Verified live CUDA hybrid retrieval and matching Chroma/Neo4j paper counts before the Phase 3 merge.

After teammate evaluation work was merged, the repository reached 123 passing tests. The multi-topic ingestion and resume guards increased the current suite to 127. Those later tests are not claimed as part of the original 39-test core handoff.

## Verification Snapshots

Original core handoff:

- Cleaned Semantic Scholar papers: 8,850
- NER papers: 8,850
- Chroma vectors: 8,850
- Neo4j papers: 8,850
- Real citation relationships: 7,203
- Core handoff tests: 39 passed

Current broad-CSE build:

- Unique raw Semantic Scholar papers: 52,822
- Cleaned, NER, Chroma, and Neo4j papers: 47,619 each
- Neo4j authors: 145,957
- Neo4j concepts: 195,252
- Real citation relationships: 22,370
- Missing original IDs after expansion: 0
- Current merged repository tests: 127 passed
- Latest recorded provenance smoke test: 5/5 accepted claims and 9/9 valid passage references

## Current Handoff State

The abstract-based product implementation is complete enough for a working capstone demo. The retrieval-evaluation infrastructure is also present, but the existing 1,329 relevance judgments were generated for the earlier corpus and now require candidate-pool refresh plus human review.

The highest-priority remaining work is:

1. Refresh the retrieval candidate pool for the expanded corpus, then human-review and freeze the benchmark.
2. Add BM25 and conventional matched-context RAG baselines.
3. Evaluate claim entailment, contradiction quality, and hypothesis quality.
4. Compare scientific NER, embedding, and LLM alternatives under fixed evaluation.
5. Run measured scaling and operational benchmarks.
6. Add evaluation reporting to the UI and live-service checks to CI.

PDF ingestion and full-text section provenance remain intentionally deferred.
