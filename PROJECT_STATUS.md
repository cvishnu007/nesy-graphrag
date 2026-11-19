# NeSy-GraphRAG Project Status

Last reconciled: November 15, 2026 (store counts below are from November 1). For a complete new-chat handoff, read [AGENT_HANDOFF.md](AGENT_HANDOFF.md) first.

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

## November 15, 2026 Update

- **Judge validated on human grades.** One annotator graded 200 pooled pairs. The frozen 3B labels agree at κ 0.17 (batch-position bias). A local 7B judge, one paper per call, reaches held-out κ 0.75; Qwen3.8-27B on Groq reaches 0.88 but the free tier is too small for full pools.
- **Retrieval, validated labels.** On 44 independent topic-stratified queries, the selected BGE + BM25 + graph system scores NDCG@10 0.644 vs 0.287 for the original production system (p < 0.0001). The gain comes from BGE and BM25 fusion; graph expansion adds a non-significant +0.023. On the regraded GNN benchmark: 0.813 vs 0.349.
- **Hypotheses and contradictions, human round 1 (negative).** Structural-hole hypotheses were less relevant than a vector baseline (3.27 vs 4.20 of 5, p = 0.002). No real contradictions were found among 56 top NLI-screened pairs. Revised methods (problem-bridge hypotheses; citation-grounded contradiction index with a 7B verdict filter) await round 2 human ratings on held-out queries.
- **Citation graph densified.** Reference lists for 46,708 papers (from 10,899) via Semantic Scholar and OpenAlex; opt-in `CITES_ADDED`, `COUPLED_WITH`, and `ExternalPaper` layers. Isolated papers fell from 71.4% to 2.4%.
- **Application now uses the validated components:** the local 7B model, citation-index contradictions, the method–task hypothesis ranker (RESULTS §4d), and a new Related Work mode. The reasoning switches await round 2 confirmation; see AGENT_HANDOFF.md.
- **Time-sliced discovery benchmark:** graph signals nearly double average precision over text similarity when predicting 2023–2025 method–task combinations from pre-2023 data (RESULTS §4c).
- **Where the graph helps.** The densified graph does not improve topical retrieval (test −0.003 vs no graph) but adds +0.068 Recall@20 (28% relative, p < 0.0001) on leakage-controlled citation recommendation. Near-duplicate collapse is enabled in `.env` (no pair co-occurs in any benchmark top 10).
- 386 tests pass. Tables: [docs/RESULTS.md](docs/RESULTS.md). Human tasks: [evaluation/human_review/README.md](evaluation/human_review/README.md).

## November 14, 2026 Update

- Public-benchmark evaluation (SciDocs 1,000 queries; LitSearch 597 queries): BGE-base with BM25 fusion is the strongest retriever; all scientific embedding models tested (SPECTER, SPECTER2 base, SciNCL, OpenScholar) are significantly weaker.
- The application now uses BGE + BM25 + graph retrieval (project test NDCG@10 0.619 vs 0.449 on the frozen 3B labels; superseded by the Nov 15 validated-label results).
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

- Second human rater for hypotheses and contradictions. (Relevance: done on a 50-pair subset, human–human κ 0.91.)
- Round 2 human ratings of the revised hypothesis and contradiction methods.
- Retrieval with the densified citation layers (a new variant; dev selection, pool extension, then test once).
- Application-mode live generation checks with the 7B verdict filter.
- Deduplication of near-duplicate papers; full-text ingestion remains out of scope.

## Key Evidence Boundaries

- Project relevance labels are LLM-judge labels validated against one human annotator, not human ground truth.
- SciFact is biomedical; in-domain contradiction quality rests on the human round-1 labels (negative) and the pending round 2.
- Graph expansion has not been shown to improve retrieval over BGE + BM25. Densification is measured as coverage, not yet as retrieval quality.
- The GNN pool was frozen on a 47,627-paper corpus; the stratified benchmark runs on the live 47,619-paper stores.

## Next Priority

Collect the second rater and round 2 ratings (instructions in [evaluation/human_review/README.md](evaluation/human_review/README.md)), score them with `src.evaluation.human_ratings`, then evaluate retrieval on the densified graph with the dev/test protocol. Commands and safety rules: [AGENT_HANDOFF.md](AGENT_HANDOFF.md).
