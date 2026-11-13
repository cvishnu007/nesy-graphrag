# NeSy-GraphRAG

NeSy-GraphRAG is a research-assistant prototype for automated literature review and hypothesis generation. It combines neural retrieval with symbolic graph validation and sentence-level provenance so generated review claims remain traceable to verified abstract evidence.

**Start here:** [AGENT_HANDOFF.md](AGENT_HANDOFF.md) is the canonical continuation document for a new developer or chat. It states the Phase 1 scope, live data/store snapshot, completed work, incomplete evaluation, safety rules, and exact next commands. [PROJECT_STATUS.md](PROJECT_STATUS.md) is the short current-status view; [SETUP.md](SETUP.md) is the clone-to-running-environment guide.

Local Ollama generation, BM25, matched-context RAG comparison, and portable reasoning reproduction are implemented. See the [local evaluation runbook](docs/LOCAL_EVALUATION.md) for the September 13 work and reproducible commands. Current research work focuses on independent judgment review, generated-output evaluation, controlled experiments, and measured scaling. PDF ingestion is not part of the current scope.

## What It Does

- Retrieves relevant scientific papers with SPECTER embeddings and ChromaDB.
- Expands results through a Neo4j citation/concept graph.
- Filters citation neighbours by query relevance before rank fusion.
- Validates paper IDs against Neo4j before sending context to the LLM.
- Generates literature reviews through local Ollama (default) or configurable Groq hosting with deterministic claim-to-passage citations.
- Blocks claims with missing, malformed, or fabricated passage IDs and retains them for audit.
- Optionally applies local NLI after passage-ID validation and retains semantically rejected claims for audit.
- Ranks contradiction candidates by normalized concept overlap and parses structured, confidence-gated verdicts.
- Ranks structural-hole hypotheses by graph evidence and validates feasibility, support, and missing evidence.
- Computes evaluation metrics for trustworthiness, graph contribution, temporal diversity, reasoning depth, and hypothesis novelty.
- Evaluates vector, graph-only, and two-way hybrid retrieval on a frozen dev/test query set.
- Provides a Streamlit UI for interactive use.

## Current Status

**November 14 update:** retrieval now uses BGE-base with BM25 fusion and graph expansion (project test NDCG@10 0.619 vs 0.449 before, p = 0.004; strongest method on the SciDocs and LitSearch public benchmarks). Concepts now come from a SciBERT model fine-tuned on SciERC (typed F1 0.639). Contradiction candidates use sentence-level NLI with evidence. See [docs/RESULTS.md](docs/RESULTS.md).

**November 1 reconciliation:** the current local corpus and stores contain 47,619 clean/NER/Chroma/Neo4j papers with 22,370 citation edges. Retrieval grading is still 778/1,145 complete; final expanded scoring and matched-RAG runs remain unfinished. See [AGENT_HANDOFF.md](AGENT_HANDOFF.md) for the authoritative handoff rather than historical contributor logs.


See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the single source of truth on implemented capabilities, missing work, limitations, and future priorities.

Short version:

- ArXiv and Semantic Scholar ingestion are implemented.
- Resumable multi-topic Semantic Scholar ingestion preserves existing records, deduplicates globally by paper ID, and checkpoints every completed topic.
- Local ArXiv and S2 data files exist.
- ChromaDB indexing is implemented and local collections exist.
- Neo4j graph loading code is implemented.
- Literature review, contradiction, hypothesis, metrics, and baseline comparison modules exist.
- The reconciled local snapshot contains 52,822 raw and 47,619 clean papers. Historical logs may contain different September 13 counts from a separate machine/run; use [AGENT_HANDOFF.md](AGENT_HANDOFF.md) for the current state.
- Hybrid rank fusion is implemented and returns neural, symbolic, and overlapping results.
- Automatic CUDA/MPS/CPU selection and CPU worker controls are implemented.
- CUDA embedding inference is verified on the local RTX 3050 with PyTorch `2.12.1+cu126`.
- Retrieval diagnostics explain ranks, citation degree, source mix, and cutoff decisions.
- Structured contradiction scoring and exact verdict parsing are implemented and tested.
- A provisional 20-query retrieval benchmark, 6/14 dev/test split, 1,329 machine-assisted judgments, IR metrics, graph-only baseline, two-way NeSy ablation, and significance analysis are implemented under `src/evaluation/`. Its recorded scores predate the broad-CSE rebuild and must not be reported as results for the expanded corpus.
- The evaluation-only hybrid scored `0.3676` NDCG@10 versus `0.3643` for vector-only, but the paired randomization p-value is `1.0`; this is not a statistically established improvement.
- Production graph filtering improved NDCG@10 from `0.2678` to `0.3617`, close to the vector-only reference at `0.3643`.
- AI-reference reasoning benchmarks, annotation tooling, semantic-support evaluation, and controlled NER/embedding/LLM comparison runners are integrated. Their labels are not human ground truth.
- The automated pytest suite covers retrieval, multi-topic ingestion, resume safety, evaluation, claim provenance, semantic support, model comparisons, configuration guards, contradiction verdicts, Neo4j failures, and hypothesis validation.
- Evidence-ranked hypothesis validation is implemented; weak/invalid generations are retained separately for audit.
- Sentence-level claim provenance is implemented; the latest live check accepted 5/5 claims with 9/9 valid passage citations.
- Production citation expansion now removes weak graph neighbours before fusion; the evaluation framework remains separate from the application pipeline.

Not yet implemented or validated:

- human review of the current machine-assisted relevance judgments before publication-level claims
- human-reviewed contradiction, claim-support, and hypothesis benchmark datasets; current reasoning benchmarks are AI-reference only
- rule-based contradiction and other task-specific baselines
- validated scientific NER, embedding, and LLM comparisons on refreshed human-reviewed references
- human validation and calibration of semantic entailment decisions
- broader component ablations and repeated stochastic model runs
- controlled 10K/50K/100K scaling benchmarks and evaluation-focused UI reporting

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the complete implementation inventory, limitations, and ordered future roadmap.

The latest teammate integration also includes a self-contained audit package under `teammate2/`. Its imported live-check evidence used the older 8,850-paper stores, and its frozen contradiction, claim-support, and hypothesis labels were generated by repeated AI annotation passes. Reproduce it as engineering evidence, not as human-validated scientific ground truth.

## Project Structure

```text
nesy-graphrag/
|-- app/streamlit_app.py      Streamlit UI for review, contradiction, and hypothesis modes
|-- src/
|   |-- ingestion/            Semantic Scholar (default) and arXiv fetchers, cleaning, spaCy NER
|   |-- storage/              Chroma/SPECTER index and Neo4j graph loader/driver
|   |-- pipeline/             retrieval, review, contradiction, hypothesis, provenance,
|   |                         prompts, verdicts, validator, metrics, orchestrator
|   |-- evaluation/           retrievers/ (BM25, graph-only, evaluation hybrid), IR metrics,
|   |                         significance, frozen-pool grading (refresh_retrieval),
|   |                         matched_rag, reasoning benchmarks/runners, annotation tooling,
|   |                         semantic support, SciFact, model comparisons, audits
|   `-- utils/                config, compute, provider-neutral llm_client, groq_client
|-- evaluation/benchmarks/    retrieval queries and judgments, SciFact support benchmark
|-- results/                  retrieval, reasoning, and local verification artifacts
|-- teammate2/                self-contained AI-reference reasoning evidence package
|-- tests/                    pytest suite (306 passing at the last full run)
|-- docs/                     evaluation runbook, design specs, and plans
|-- AGENT_HANDOFF.md          canonical continuation handoff
|-- PROJECT_STATUS.md, SETUP.md, README.md
`-- requirements.txt, pytest.ini, .env.example
```

Generated data lives under `data/`, which is ignored by Git.

## Setup

Follow [SETUP.md](SETUP.md) for the verified Windows installation, local Neo4j configuration, CUDA/CPU choices, environment variables, from-scratch data build, validation commands, clean rebuild procedure, and troubleshooting.

## Requirements

- Python 3.11, 64-bit for the primary CUDA setup
- Python 3.13.14 also verified for the CPU evaluation environment
- A dedicated local Neo4j database
- Ollama with an installed model, or a Groq API key for hosted generation
- Semantic Scholar API key for S2 ingestion

Dependencies are pinned to the tested environment. From the repository root:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m spacy download en_core_web_sm
```

Run the automated test suite:

```powershell
.\venv\Scripts\python.exe -m pytest
```

Pytest configuration lives in `pytest.ini`. Shared fixtures belong in `tests/conftest.py`; tests that require live services must use the registered `integration` marker.

### NVIDIA GPU Setup

The code selects CUDA automatically when the installed PyTorch build supports it. On Windows, install a CUDA build into the existing virtual environment before installing the remaining requirements. For the detected RTX 3050, use the official CUDA 12.6 wheels:

```powershell
.\venv\Scripts\python.exe -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The CUDA wheel is approximately 2.6 GB. Let the first command complete before running the requirements command.

The current local environment has this CUDA build installed and verified. A SPECTER query automatically selected the RTX 3050 and used approximately 429 MB allocated VRAM.

Verify GPU access:

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`EMBEDDING_DEVICE=auto` uses CUDA first, then Apple MPS, then CPU. SPECTER uses `EMBEDDING_BATCH_SIZE=16` by default to stay within a 4 GB GPU. The current spaCy model uses GPU only when spaCy's GPU backend is installed; otherwise NER automatically uses parallel CPU workers.

## Configuration

Create a local `.env` file in the project root:

```env
DATA_SOURCE=s2

NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_ALLOW_RESET=false

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:3b-instruct
# GROQ_API_KEY is needed only for LLM_PROVIDER=groq
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key

S2_QUERIES=graph neural networks;artificial intelligence and machine learning;cybersecurity;computer networks;databases;software engineering;cloud computing;natural language processing;computer vision;data science;operating systems
S2_LIMIT_PER_QUERY=5000
S2_INCLUDE_EXISTING=true

EMBEDDING_MODEL=allenai-specter
EMBEDDING_DEVICE=auto
EMBEDDING_BATCH_SIZE=16
LLM_MODEL=openai/gpt-oss-120b
LLM_MODEL_FALLBACK=llama-3.1-8b-instant

EVALUATION_START_YEAR=2020
EVALUATION_END_YEAR=2025

CONTRADICTION_MIN_SHARED_CONCEPTS=2
CONTRADICTION_MIN_CONCEPT_JACCARD=0.10
CONTRADICTION_MIN_CONFIDENCE=0.70

HYPOTHESIS_MIN_SHARED_CONCEPTS=2
HYPOTHESIS_MIN_QUERY_SUPPORT=2

GRAPH_HIGH_SEMANTIC_THRESHOLD=0.85
GRAPH_SEMANTIC_FLOOR=0.75
GRAPH_MIN_QUERY_TERM_COVERAGE=0.75
GRAPH_STRONG_CONNECTIONS=10

EVALUATION_HYBRID_VECTOR_WEIGHT=16.0
EVALUATION_HYBRID_GRAPH_WEIGHT=1.0
GRAPH_ONLY_CANDIDATE_LIMIT=100
```

Useful defaults are defined in `src/utils/config.py`.

Production retrieval follows this flow:

```text
Vector retrieval -> citation expansion -> graph relevance filter -> RRF fusion -> output
```

A graph candidate is kept when its stored SPECTER embedding has cosine similarity
of at least `0.85`, or when it passes the semantic floor, covers at least 75% of
meaningful query terms, and connects to at least 10 distinct vector seed papers.
The stricter connection requirement is intentional: two-hop citation paths made
the earlier value of two too permissive on the development queries.

Keep `NEO4J_ALLOW_RESET=false` in `.env`. Enable it only for the single command that rebuilds a dedicated project database, because graph loading clears the configured database.

## Pipeline

Run the data and graph setup in this order:

```powershell
.\venv\Scripts\python.exe -m src.ingestion.run_ingestion
.\venv\Scripts\python.exe -m src.ingestion.ner_extractor
.\venv\Scripts\python.exe -m src.storage.chroma_store
.\venv\Scripts\python.exe -m src.storage.neo4j_store
```

Run the full pipeline smoke test:

```powershell
.\venv\Scripts\python.exe -m src.pipeline.orchestrator
```

Run the baseline comparison:

```powershell
.\venv\Scripts\python.exe -m src.pipeline.baseline_harness
```

Explain a hybrid retrieval result without calling the LLM:

```powershell
.\venv\Scripts\python.exe -m src.pipeline.retrieval "graph neural networks for node classification" --top-k 10
```

## Retrieval Evaluation

The historical retrieval benchmark is version `0.2-draft` and contains 20 frozen queries, split into 6 development and 14 test queries. Its 1,329 relevance labels are provisional, were created against the earlier 8,850-paper corpus, and must be regenerated or extended for each expanded corpus before new scores are reported. Use the frozen-pool workflow in [docs/LOCAL_EVALUATION.md](docs/LOCAL_EVALUATION.md).

Run the vector, graph-only, and evaluation-only hybrid comparison with Neo4j and Chroma available:

```powershell
.\venv\Scripts\python.exe -m src.evaluation.retrieval_runner --split test --top-k 20 --output-dir results/retrieval/evaluation --overwrite
```

Run paired significance analysis:

```powershell
.\venv\Scripts\python.exe -m src.evaluation.significance results/retrieval/evaluation/per_query_metrics.csv --challenger hybrid --reference vector --metric ndcg@10 --output results/retrieval/evaluation/significance.json
```

Do not tune using the 14-query test split. Review `evaluation/benchmarks/retrieval_judgments_draft.csv`, finalize the labels, and freeze a new benchmark version before treating the results as final.

## App

Start the Streamlit UI:

```powershell
.\venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The app supports:

- Literature Review
- Claim evidence inspection and unsupported-claim audits
- Contradiction Detection
- Hypothesis Generation

## Metrics

Implemented in `src/pipeline/metrics.py`:

- `TS`: Trustworthiness Score
- `NBR`: NeSy Boost Ratio
- `ATD`: Answer Temporal Diversity
- `RDI`: Reasoning Depth Index
- `HNS`: Hypothesis Novelty Score

These are prototype diagnostics, not established scientific benchmarks. TS uses valid passage citations and claim coverage, NBR measures graph participation rather than relevance, and HNS measures normalized graph-path distance rather than expert-assessed novelty. See [AGENT_HANDOFF.md](AGENT_HANDOFF.md) for limitations and evidence boundaries.

The review UI displays only `TS` and `ATD`, which are computed from that review
run. `NBR`, `RDI`, and `HNS` remain backend diagnostics for retrieval analysis
or combined review/contradiction/hypothesis experiments.

## Scope Boundaries

- The reconciled corpus is 47,619 cleaned Semantic Scholar records collected through 11 broad CSE topic queries, primarily abstracts and metadata. It is not a balanced taxonomy, a relevance-labeled corpus, or the million-scale full-text corpus proposed in the Phase 1 report.
- The teammate’s historical expanded graph had 22,370 real citation edges across 47,619 papers. Topic smoke tests found useful vector results, but strict graph filtering retained a result for only one of 11 representative queries; scaling alone has not solved sparse graph contribution.
- The project uses pretrained SPECTER embeddings and local Ollama `qwen2.5:3b-instruct` by default, with optional Groq hosting; it does not train or fine-tune a local reasoning model.
- Review claims are traceable to deterministic sentence IDs from verified abstracts. Passage existence is enforced; semantic entailment quality is not yet benchmarked.
- PDF ingestion and full-text section provenance remain intentionally deferred for this phase.

## Development Notes

- Keep credentials in `.env`; do not commit them.
- Keep generated datasets and ChromaDB files under `data/`.
- Rebuild the S2 Chroma collection before final evaluation if its vector count does not match the current cleaned S2 dataset.
- Resource usage changes by stage: embedding can use the GPU, NER uses GPU when supported or parallel CPU otherwise, and API/Neo4j stages can be network or database bound.
- Pytest unit tests use small in-memory driver doubles only to isolate failure/guard behavior; end-to-end checks use the real local Neo4j graph and Chroma index.
- Update [AGENT_HANDOFF.md](AGENT_HANDOFF.md) and [PROJECT_STATUS.md](PROJECT_STATUS.md) together after material implementation, runtime, evaluation, or scope changes.
