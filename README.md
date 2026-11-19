# NeSy-GraphRAG

A research assistant for computer-science literature that combines text retrieval with citation and scientific-entity graphs, and an evaluation of where the graphs actually help. Final-year capstone project; paper draft in [docs/PAPER_DRAFT.md](docs/PAPER_DRAFT.md).

**To run it:** [docs/RUN_FROM_SCRATCH.md](docs/RUN_FROM_SCRATCH.md) covers everything from a fresh machine to the running app and the reproduced results.

## What it does

The app runs over 47,619 computer-science abstracts (Semantic Scholar, 2020–2025, 11 topics) with a local language model (`qwen2.5:7b-instruct` via Ollama):

| Mode | What it does | How it is grounded |
|---|---|---|
| 📚 **Literature Review** | Writes a cited summary for a research question | Every claim must cite a sentence from a retrieved abstract; claims with invalid citations are blocked and listed |
| ⚡ **Contradiction Detection** | Finds papers that dispute a paper they cite | Only pairs linked by a real citation; conflicting sentences shown side by side; LLM verdict filter |
| 💡 **Hypothesis Generation** | Suggests method → task combinations not yet tried | Entity-graph and citation evidence, plus an example paper |
| 🔗 **Related Work** | Suggests papers to cite, with an interactive citation map | Co-citation records: the similar papers whose reference lists include each suggestion |

Example queries for each mode: [docs/DEMO_QUERIES.md](docs/DEMO_QUERIES.md).

## Main findings

**Why "NeSy":** neural models (embeddings, the LLM, NLI, the entity tagger) propose; a typed citation and entity graph and explicit rules (passage-ID provenance checks, citation-link and entity constraints) restrict and verify what is shown. See [PAPER_DRAFT §3.1](docs/PAPER_DRAFT.md).

Measured with frozen dev/test splits, an LLM relevance judge validated against human grades (on GNN-topic pairs; a check on the other topics is prepared), two human raters, and paired significance tests. Full tables: [docs/RESULTS.md](docs/RESULTS.md).

The knowledge graph (Neo4j: papers, authors, 323k scientific entities, and original plus densified citation links) is the backbone of the system. Every mode reads from it. The evaluation measures, task by task, how much the graph adds on top of strong text retrieval:

| Task | Result | Role of the graph |
|---|---|---|
| Citation recommendation (our corpus) | Co-citation adds **+28%** Recall@20 over text retrieval (p < 0.0001) | **Main source of the gain** |
| Citation recommendation (public DBLP, PubMed) | Co-citation raises Recall@20 on top of BGE, SPECTER2 and SciNCL (+0.07–0.08 on DBLP). With SPECTER2 (itself pretrained on citations) and no training on DBLP, it comes close to the trained Citeomatic system (F1@20 0.300 vs 0.303; MRR 0.708 vs 0.689). The expansion follows Strohman et al. (2007) and Citeomatic's NNSelect; the new evidence is that it still adds to modern encoders | **Adds to modern embeddings, including citation-trained ones**; strongest in CS |
| Contradiction detection (human-labelled) | Citation grounding raises contradiction-or-tension precision from 9% (5/56) to 56% (5/9; 95% CI 27–81%; one rater) | **Makes the mode work**: candidate pairs are real citation links |
| Time-sliced discovery | Graph signals nearly double average precision for predicting which method–task pairs researchers adopt later | **Main predictive signal** |
| Topical retrieval (44 queries) | BGE + BM25 + graph NDCG@10 **0.644**, vs 0.287 for the original system; a LightRAG-style GraphRAG baseline scores 0.396 | Text retrieval already saturates this task. The graph keeps quality level (+0.023) and supplies the citation and ID checks |
| Hypothesis candidates (human-rated) | Graph-selected candidate papers are rated below vector neighbours in two rounds | This led to the method–task ranker now used in the app, which is based on the discovery signal above |
| Review answer grounding | 41.8% vs 38.9% claims fully supported by the cited passage | Grounding comes from claim-level citation checks, which every answer passes through |

In short, the graph matters most where the question is how papers relate to each other (who cites, disputes or will combine what), and least for plain topic search, where text retrieval is already strong.

## Repository layout

```text
app/                 Streamlit app (streamlit_app.py) and the citation-map view (graph_view.py)
src/ingestion/       Semantic Scholar download, cleaning, spaCy and SciERC entity extraction, citation densification
src/storage/         Chroma vector index, Neo4j graph loader, graph enrichment
src/pipeline/        Retrieval, review, contradiction, hypothesis and related-work pipelines; provenance checks
src/evaluation/      Benchmarks, LLM judge, human-rating scorers, baselines (LightRAG-style, DBLP/PubMed), reproduce.py
evaluation/          Benchmark queries, human review sheets and instructions
results/             Saved rankings, labels and summaries behind every reported number
docs/                Run guide, results, paper draft, evaluation protocol, demo queries
tests/               pytest suite (406 tests; no databases or models needed)
```

Generated data (`data/`), the fine-tuned entity model (`models/`) and `.env` are not committed.

## Quick reference

```powershell
.\venv\Scripts\python.exe -m pytest -q                                        # tests
.\venv\Scripts\python.exe -m src.evaluation.reproduce                         # check 23 headline numbers
.\venv\Scripts\python.exe -m streamlit run app\streamlit_app.py --server.address localhost   # app
```

## Documentation

| File | Use it for |
|---|---|
| [docs/RUN_FROM_SCRATCH.md](docs/RUN_FROM_SCRATCH.md) | Installing, building the data, running the app, reproducing results |
| [docs/DEMO_QUERIES.md](docs/DEMO_QUERIES.md) | Queries that work well in each mode |
| [docs/RESULTS.md](docs/RESULTS.md) | Every result, with the files it comes from |
| [docs/PAPER_DRAFT.md](docs/PAPER_DRAFT.md) | The paper draft |
| [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md) | Metrics, dev/test rules, recorded selections |
| [evaluation/human_review/README.md](evaluation/human_review/README.md) | Human rating tasks and how they were scored |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Short current status |
| [AGENT_HANDOFF.md](AGENT_HANDOFF.md) | Detailed state, decisions and safety rules for continuing development |

## Scope and limits

- Abstracts and metadata only; no PDF or full-text processing.
- Only one model is trained: a SciBERT entity tagger fine-tuned on SciERC (typed F1 0.64). Retrieval and generation use pretrained models.
- Relevance labels come from a 7B LLM judge validated against human grades on GNN-topic pairs (held-out κ 0.75; two humans agree at κ 0.91). Its accuracy on the other ten topics is not yet measured.
- Rater 1 on every human task built the system (sheets were blind to method); round 2 has one rater.
- Outputs are research suggestions to check, not verified scientific conclusions.

## Team

Capstone project by Vishnu, Chinmay, Devesh and Gurleen. Historical work logs: `VISHNU.md`, `Chinmay.md`, `devesh.md`.
