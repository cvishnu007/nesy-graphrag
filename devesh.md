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
- Evaluated the application's Vector + citation-graph Hybrid against standard baselines.

## Relevance scale

- `0`: not relevant to the complete query.
- `1`: meaningfully related, but not the paper's central focus.
- `2`: directly relevant; the query topic is central to the paper.

The benchmark remains marked `judgments_pending_human_review`. The current judgments are suitable for development and provisional evaluation; a teammate should review a sample before a final publication-level claim.

## Hybrid design

The evaluated Hybrid is the same two-stage retrieval design used by the application: Vector retrieval followed by Neo4j citation expansion.

Weights were selected using only the 6 development queries:

- Vector weight: `16.0`
- Citation-graph weight: `1.0`
- Reciprocal-rank constant: `RRF_K=60`

The weights were frozen before evaluation on the 14 test queries.

## Final test results

| Method | NDCG@10 | Recall@10 | MAP | MRR |
|---|---:|---:|---:|---:|
| BM25 | **0.7270** | **0.3125** | **0.5896** | **0.9643** |
| Vector + citation graph Hybrid | 0.3676 | 0.1440 | 0.1885 | 0.6500 |
| Vector | 0.3643 | 0.1440 | 0.1886 | 0.6500 |
| Graph-only | 0.2399 | 0.0841 | 0.1125 | 0.5133 |

Hybrid versus Vector on the 14 test queries:

- Wins: 1
- Ties: 13
- Losses: 0
- Mean NDCG@10 improvement: `+0.0033`, approximately `+0.91%`
- Paired bootstrap 95% interval: `[0.0000, 0.0099]`
- Exact two-sided randomization p-value: `1.0000`

The accurate conclusion is that the two-way Hybrid slightly improves Vector retrieval on this test set, but the improvement is not statistically significant. BM25 remains the strongest overall baseline. The citation graph has limited ranking impact at the frozen `16:1` weight and primarily reranks papers already found by Vector retrieval.

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
- `src/evaluation/retrievers/bm25_retrieval.py`: lexical baseline.
- `src/evaluation/retrievers/graph_only_retrieval.py`: concept-graph baseline.
- `src/pipeline/retrieval.py`: application Vector + citation-graph Hybrid.
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

Run the paired Hybrid-versus-Vector analysis:

```bat
.\venv\Scripts\python.exe -m src.evaluation.significance results\retrieval\evaluation_tuned_test\per_query_metrics.csv --challenger hybrid --reference vector --output results\retrieval\evaluation_tuned_test\significance.json
```

## Branch

Work is prepared on the `retrieval_evaluation` branch for review through a pull request before merging into `master`.
