# NeSy-GraphRAG Project Status

Last reconciled: November 1, 2026. For a complete new-chat handoff, read [AGENT_HANDOFF.md](AGENT_HANDOFF.md) first.

## Current State

The Phase 1 scope is an evidence-grounded research-assistant prototype: literature retrieval, provenance-aware review generation, cross-paper contradiction candidates, hypothesis generation, and rigorous evaluation. The system operates on paper metadata and abstracts. PDF/full-text ingestion and model training are intentionally outside the current scope.

The live local data/store snapshot was checked on November 1:

| Asset | Count/status |
|---|---:|
| Raw Semantic Scholar records | 52,822 |
| Clean papers | 47,619 |
| NER records | 47,619 |
| Chroma vectors | 47,619 |
| Neo4j Paper nodes | 47,619 |
| Neo4j citation edges | 22,370 |
| Neo4j | Running at `127.0.0.1:7687` during verification |
| Ollama / Streamlit | Stopped during verification |

The repository is on `phase3` at `8f2eaa2`, four commits ahead of `origin/phase3`, and matches `origin/feat/local-evaluation`.

## Implemented

- Resumable Semantic Scholar ingestion, global deduplication, cleaning, spaCy NER, SPECTER/Chroma indexing, and Neo4j graph loading.
- Provider-neutral generation: local Ollama by default and optional Groq.
- Vector retrieval, citation-graph expansion, graph relevance filtering, weighted reciprocal-rank fusion, and retrieval diagnostics.
- Sentence-level claim provenance; structurally invalid citations are withheld and audited.
- Structured contradiction verdicts and evidence-ranked hypothesis generation.
- Streamlit UI for review, contradiction, and hypothesis workflows.
- BM25/vector/graph/hybrid baselines; artifact auditing; packaged reasoning reproduction; SciFact and support-calibration tooling.
- Latest recorded full verification: 306 tests passed, one non-fatal upstream spaCy/Click deprecation warning.

## November 14, 2026 Update

- Public-benchmark evaluation (SciDocs 1,000 queries; LitSearch 597 queries): BGE-base with BM25 fusion is the strongest retriever; all scientific embedding models tested (SPECTER, SPECTER2 base, SciNCL, OpenScholar) are significantly weaker.
- The application now uses BGE + BM25 + graph retrieval (project test NDCG@10 0.619 vs 0.449, p = 0.004).
- Entity extraction now uses SciBERT fine-tuned on SciERC (test typed F1 0.639; previous spaCy span F1 0.270), stored as a separate `SciEntity` graph.
- Contradiction candidates now come from NLI over finding sentences, with structured verdicts.
- 356 tests pass. Full tables: [docs/RESULTS.md](docs/RESULTS.md).

## November 13, 2026 Update

- Retrieval grading completed (1,145/1,145) and the frozen benchmark was scored. Test NDCG@10: BM25 0.529, production 0.455, vector 0.435, graph-only 0.413; no difference is significant with 14 queries.
- Generic-concept flags and 238,095 `SIMILAR_TO` kNN edges were added to Neo4j (additive). A title+abstract SPECTER index was built in `s2_papers_specter_ta`.
- New opt-in retrieval, contradiction, hypothesis, and HNS options; dev ablation and pool-extension tooling; a topic-stratified 44-query draft; and [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).
- The dev-selected configuration did not beat BM25 on test (0.476 vs 0.529), so production defaults are unchanged. Details are in [AGENT_HANDOFF.md](AGENT_HANDOFF.md).
- 334 tests pass.

## Incomplete or Not Yet Validated

- Matched-budget RAG outputs have not been produced.
- The 20-query benchmark covers only graph neural networks; the stratified 44-query draft needs team review, pooling, and grading.
- Application-mode live generation checks need a new Ollama run on the current stores.
- Relevance judgments, contradiction references, claim support, and hypotheses are not human-reviewed ground truth.
- Controlled scientific NER/embedding/LLM comparisons, human semantic-entailment calibration, repeated runs/confidence intervals, and controlled scaling remain future work.

## Key Evidence Boundaries

- Historical 8,850-paper hybrid NDCG@10 was 0.3676 versus 0.3643 vector-only, with `p=1.0`; it did not establish hybrid superiority and is not an expanded-corpus result.
- The current citation graph is sparse: 22,370 edges over 47,619 papers. Scaling alone has not established useful graph contribution.
- SciFact NLI is a separate biomedical claim/evidence benchmark, not a graph-retrieval or cross-paper-contradiction validation.
- The paused expanded retrieval pool was frozen on a 47,627-paper corpus, not the current 47,619-paper stores. Its scores and live matched-RAG outputs belong to different corpora; see [AGENT_HANDOFF.md](AGENT_HANDOFF.md).

## Next Priority

Start Ollama, resume the frozen retrieval-grading checkpoint, score only after completion, then obtain independent human review before making research-performance claims. Exact commands and safety rules are in [AGENT_HANDOFF.md](AGENT_HANDOFF.md) and [docs/LOCAL_EVALUATION.md](docs/LOCAL_EVALUATION.md).
