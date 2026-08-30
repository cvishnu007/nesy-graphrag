# Devesh — Retrieval and Evaluation Work

## Scope completed

This branch completes the retrieval benchmarking and evaluation work for the NeSy GraphRAG project.

- Prepared and verified the Python 3.13 CPU environment.
- Verified 8,850 papers in both the cleaned Semantic Scholar data and ChromaDB.
- Built the Neo4j graph with 8,850 papers, 27,655 authors, 42,937 concepts, and 7,203 citation edges.
- Added a frozen 20-query retrieval benchmark with 6 development and 14 test queries.
- Added standard retrieval metrics: Precision, Recall, Hit Rate, MRR, MAP, NDCG, and unjudged rate.
- Added strict benchmark loading and validation.
- Added deterministic BM25 and Graph-only baselines.
- Added method-hidden candidate pooling across BM25, Vector, Graph, and Hybrid retrieval.
- Prepared 1,329 query-paper relevance judgments using the 0/1/2 relevance scale.
- Added judgment finalization, four-method evaluation, and paired significance analysis.
- Reworked Hybrid retrieval as weighted BM25 + Vector + Graph-only reciprocal-rank fusion.

## Relevance scale

- `0`: not relevant to the complete query.
- `1`: meaningfully related, but not the paper's central focus.
- `2`: directly relevant; the query topic is central to the paper.

The benchmark remains marked `judgments_pending_human_review`. The current judgments are suitable for development and provisional evaluation; a teammate should review a sample before a final publication-level claim.

## Hybrid design

The original Hybrid used Vector retrieval followed by citation expansion. Citation proximity often promoted connected but topically weaker papers.

The tuned Hybrid combines three complementary rankings:

- BM25 weight: `2.0`
- Vector weight: `1.0`
- Graph-only weight: `1.0`
- Reciprocal-rank constant: `RRF_K=60`

Weights were selected using only the 6 development queries. They were frozen before evaluation on the 14 test queries.

## Final test results

| Method | NDCG@10 | Recall@10 | MAP | MRR |
|---|---:|---:|---:|---:|
| Tuned Hybrid | **0.7670** | **0.3161** | **0.5912** | **0.9643** |
| BM25 | 0.7270 | 0.3125 | 0.5896 | 0.9643 |
| Vector | 0.3643 | 0.1440 | 0.1886 | 0.6500 |
| Graph-only | 0.2399 | 0.0841 | 0.1125 | 0.5133 |

Hybrid versus BM25 on the 14 test queries:

- Wins: 6
- Ties: 5
- Losses: 3
- Mean NDCG@10 improvement: `+0.0399`, approximately `+5.5%`
- Paired bootstrap 95% interval: `[0.0080, 0.0745]`
- Exact two-sided randomization p-value: `0.0547`

The accurate conclusion is that the tuned Hybrid shows a promising improvement on this test set. The exact randomization result is borderline, so it should not be described as conclusive statistical significance.

## Important files

- `evaluation/benchmarks/retrieval_queries.json`: frozen query definitions.
- `evaluation/benchmarks/retrieval_judgments_draft.csv`: judgment worksheet and notes.
- `evaluation/benchmarks/retrieval_queries_judged.json`: judged benchmark used by the runner.
- `src/evaluation/ir_metrics.py`: retrieval metric implementations.
- `src/evaluation/benchmark_io.py`: benchmark validation.
- `src/evaluation/candidate_pool.py`: method-hidden candidate pooling.
- `src/evaluation/finalize_judgments.py`: CSV-to-benchmark conversion.
- `src/evaluation/retrieval_runner.py`: four-method evaluation runner.
- `src/evaluation/significance.py`: paired uncertainty and randomization analysis.
- `src/pipeline/bm25_retrieval.py`: lexical baseline.
- `src/pipeline/graph_only_retrieval.py`: concept-graph baseline.
- `src/pipeline/tuned_hybrid_retrieval.py`: frozen three-way Hybrid.
- `results/retrieval/evaluation_tuned_test/`: final test rankings, metrics, summary, and significance output.

## Reproduction commands

Run the full test suite:

```bat
.\venv\Scripts\python.exe -m pytest
```

Rebuild the judged benchmark from the reviewed worksheet:

```bat
.\venv\Scripts\python.exe -m src.evaluation.finalize_judgments
```

Run the frozen test evaluation:

```bat
.\venv\Scripts\python.exe -m src.evaluation.retrieval_runner --split test --output-dir results\retrieval\evaluation_tuned_test --overwrite
```

Run the paired Hybrid-versus-BM25 analysis:

```bat
.\venv\Scripts\python.exe -m src.evaluation.significance results\retrieval\evaluation_tuned_test\per_query_metrics.csv --output results\retrieval\evaluation_tuned_test\significance.json
```

## Branch

Work is prepared on the `retrieval_evaluation` branch for review through a pull request before merging into `master`.
