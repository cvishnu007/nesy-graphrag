# Devesh — Retrieval and Evaluation Work

## Scope completed

This branch completes the retrieval benchmarking and evaluation work for the NeSy GraphRAG project.

- Prepared and verified the Python 3.13 CPU environment.
- Verified 8,850 papers in both the cleaned Semantic Scholar data and ChromaDB.
- Built the Neo4j graph with 8,850 papers, 27,655 authors, 42,937 concepts, and 7,203 citation edges.
- Added a frozen 20-query retrieval benchmark with 6 development and 14 test queries.
- Added standard retrieval metrics: Precision, Recall, Hit Rate, MRR, MAP, NDCG, and unjudged rate.
- Added strict benchmark loading and validation.
- Added a Graph-only evaluation retriever.
- Added method-hidden candidate pooling across Vector, Graph, and Hybrid retrieval.
- Prepared 1,329 query-paper relevance judgments using the 0/1/2 relevance scale.
- Added judgment finalization, three-method evaluation, and paired significance analysis.
- Added an evaluation-only Vector + citation-graph Hybrid without changing the application pipeline.

## Relevance scale

- `0`: not relevant to the complete query.
- `1`: meaningfully related, but not the paper's central focus.
- `2`: directly relevant; the query topic is central to the paper.

The benchmark remains marked `judgments_pending_human_review`. The current judgments are suitable for development and provisional evaluation; a teammate should review a sample before a final publication-level claim.

## Evaluation scope

This benchmark is a scoped NeSy ablation comparing Vector, Graph-only, and Hybrid retrieval. It does not include lexical or external baselines, so these results must not be presented as proof that Hybrid is better than every possible retrieval method.

The production Streamlit pipeline remains unchanged and continues to use its original `1:1` Vector/citation-graph weights. The evaluation-only Hybrid uses the same two retrieval stages with development-tuned weights:

- Vector weight: `16.0`
- Citation-graph weight: `1.0`
- Reciprocal-rank constant: `RRF_K=60`

Weights were selected using only the 6 development queries. They were frozen before evaluation on the 14 test queries.

## Final test results

| Method | NDCG@10 | Recall@10 | MAP | MRR |
|---|---:|---:|---:|---:|
| Evaluation-only Hybrid | **0.3676** | **0.1440** | 0.1885 | **0.6500** |
| Vector | 0.3643 | 0.1440 | 0.1886 | 0.6500 |
| Graph-only | 0.2399 | 0.0841 | 0.1125 | 0.5133 |

Hybrid versus Vector on the 14 test queries:

- Wins: 1
- Ties: 13
- Losses: 0
- Mean NDCG@10 improvement: `+0.0033`, approximately `+0.91%`
- Paired bootstrap 95% interval: `[0.0000, 0.0099]`
- Exact two-sided randomization p-value: `1.0000`

The accurate conclusion is that the evaluation-only Hybrid ranks slightly above Vector on this test set, but the difference is not statistically significant. The graph has limited ranking impact at the frozen `16:1` evaluation weight.

## Important files

- `evaluation/benchmarks/retrieval_queries.json`: frozen query definitions.
- `evaluation/benchmarks/retrieval_judgments_draft.csv`: judgment worksheet and notes.
- `evaluation/benchmarks/retrieval_queries_judged.json`: judged benchmark used by the runner.
- `src/evaluation/ir_metrics.py`: retrieval metric implementations.
- `src/evaluation/benchmark_io.py`: benchmark validation.
- `src/evaluation/candidate_pool.py`: method-hidden candidate pooling.
- `src/evaluation/finalize_judgments.py`: CSV-to-benchmark conversion.
- `src/evaluation/retrieval_runner.py`: three-method evaluation runner.
- `src/evaluation/significance.py`: paired uncertainty and randomization analysis.
- `src/evaluation/retrievers/graph_only_retrieval.py`: concept-graph baseline.
- `src/evaluation/retrievers/two_way_hybrid_retrieval.py`: evaluation-only Vector + citation-graph Hybrid.
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

## Merge status

The retrieval and evaluation work was merged into `master` through pull request #5. This document records the completed implementation, reproduction commands, and provisional evaluation results.





SECONG PART
# Production Graph Relevance Improvement

## What was wrong

The main application first retrieved relevant papers from ChromaDB and then
expanded through one- or two-hop Neo4j citation links. All 20 graph neighbours
were allowed into rank fusion. A citation connection does not always mean that a
paper answers the user's query, so loosely related graph papers sometimes entered
the final result and weakened the LLM evidence.

Old flow:

```text
Vector retrieval -> citation expansion -> equal RRF fusion -> output
```

## What was changed

The application now checks graph neighbours before fusion:

```text
Vector retrieval -> citation expansion -> relevance filter -> equal RRF fusion -> output
```

For each graph candidate, the filter:

1. Reads its existing SPECTER embedding from ChromaDB.
2. Calculates cosine similarity with the cached query embedding.
3. Measures meaningful query-term coverage in its title and abstract.
4. Uses the number of distinct vector seed papers connected through the graph.
5. Keeps the paper only when it passes one of the safe rules below.

Safe rules:

- semantic similarity is at least `0.85`; or
- similarity is at least `0.75`, query-term coverage is at least `0.75`, and the
  paper connects to at least `10` distinct vector seed papers.

The existing vector and graph RRF weights remain `1.0` and `1.0`. Ingestion,
cleaning, NER, Neo4j loading, Chroma indexing, LLM prompts, provenance,
contradiction detection, hypothesis generation, and the evaluation-only
retrievers were not changed.

## Why these values were selected

Thresholds were selected using only the six development queries. The 14 test
queries were checked once after selection.

| Retrieval flow | Test NDCG@10 | Test Recall@10 | Mean graph papers kept |
|---|---:|---:|---:|
| Old unfiltered vector + graph | 0.2678 | 0.1117 | 20.00 |
| New filtered vector + graph | 0.3617 | 0.1422 | 0.79 |
| Vector only, reference result | 0.3643 | 0.1440 | 0.00 |

The new filter removes the large quality loss caused by weak graph neighbours and
keeps the graph available for strong discoveries. It does not falsely claim that
the graph always beats vector-only retrieval. The relevance judgments are still
pending human review before publication-level claims.

The earlier idea of accepting a paper connected to only two vector seeds was
tested and rejected. With two-hop citation expansion, weak candidates already had
several seed connections. A threshold of 10 was the first safe development choice
with zero unjudged results.

## Files changed

- `src/pipeline/retrieval.py`: relevance filtering, production integration, and
  clearer diagnostics.
- `src/storage/chroma_store.py`: cosine scoring using stored paper embeddings.
  The query vector is cached so vector retrieval and graph filtering do not run
  the same model encoding twice.
- `src/utils/config.py`: validated filter settings.
- `.env.example`: documented defaults for a fresh clone.
- `tests/test_retrieval.py`: filter behavior and production integration tests.
- `tests/test_chroma_scores.py`: stored-embedding scoring tests.
- `README.md`, `SETUP.md`, and `PROJECT_STATUS.md`: setup and status updates.

## Verification completed

- Python: `3.13.14`
- Full unit suite: `123 passed`
- `pip check`: no broken requirements
- Python compilation: successful
- Cleaned S2 papers: `8,850`
- Chroma papers: `8,850`
- Neo4j papers: `8,850`
- Real CITES relationships: `7,203`
- Loose example query: 20 raw graph candidates, 0 retained
- Relevant example query: 20 raw graph candidates, 1 retained
- Streamlit Literature Review: 10/10 papers verified and 5/5 claims grounded
- Streamlit Contradiction Detection: 2 pairs evaluated, no exception
- Streamlit Hypothesis Generation: 5/5 hypotheses valid, no exception

## Teammate clone and demo steps

From a fresh clone, create `.env` from `.env.example` and add private credentials.
Do not commit `.env`.

```powershell
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m spacy download en_core_web_sm
```

Start the existing Neo4j Docker container if it is already created:

```powershell
docker start nesy-neo4j
```

Verify the environment and data:

```powershell
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -c "from src.storage.chroma_store import get_collection; print(get_collection().count())"
.\venv\Scripts\python.exe -c "from src.storage.neo4j_store import get_driver; d=get_driver(); d.verify_connectivity(); print('Neo4j OK'); d.close()"
```

Inspect retrieval without spending an LLM API call:

```powershell
.\venv\Scripts\python.exe -m src.pipeline.retrieval "self-supervised graph representation learning" --top-k 10
```

Start the application:

```powershell
.\venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

For the panel, demonstrate all three modes with a clear graph-learning query.
Explain that vector retrieval supplies semantically close papers, Neo4j discovers
citation-connected candidates, the relevance filter removes weak neighbours, and
the LLM receives only the final verified evidence.

## Important limitation

A fresh clone does not contain the ignored `data/` directory or the contents of a
local Docker volume. The teammate must either receive the prepared data and
Neo4j volume separately or run the ingestion, NER, Chroma, and Neo4j build stages
from `SETUP.md`. Pushing code to GitHub alone does not transfer those databases.
