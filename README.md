# NeSy-GraphRAG

NeSy-GraphRAG is a research-assistant prototype for automated literature review and hypothesis generation. It combines neural retrieval with symbolic graph validation and sentence-level provenance so generated review claims remain traceable to verified abstract evidence.

The abstract-based core pipeline is implemented. Current work should focus on benchmark construction, standard evaluation, baseline comparisons, controlled model experiments, and measured scaling. PDF ingestion is not part of the current scope.

## What It Does

- Retrieves relevant scientific papers with SPECTER embeddings and ChromaDB.
- Expands results through a Neo4j citation/concept graph.
- Validates paper IDs against Neo4j before sending context to the LLM.
- Generates literature reviews through a configurable Groq-hosted model with deterministic claim-to-passage citations.
- Blocks claims with missing, malformed, or fabricated passage IDs and retains them for audit.
- Ranks contradiction candidates by normalized concept overlap and parses structured, confidence-gated verdicts.
- Ranks structural-hole hypotheses by graph evidence and validates feasibility, support, and missing evidence.
- Computes evaluation metrics for trustworthiness, graph contribution, temporal diversity, reasoning depth, and hypothesis novelty.
- Provides a Streamlit UI for interactive use.

## Current Status

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the single source of truth on implemented capabilities, missing work, limitations, and future priorities.

Short version:

- ArXiv and Semantic Scholar ingestion are implemented.
- Local ArXiv and S2 data files exist.
- ChromaDB indexing is implemented and local collections exist.
- Neo4j graph loading code is implemented.
- Literature review, contradiction, hypothesis, metrics, and baseline comparison modules exist.
- The clean S2 run contains 8,850 papers in both ChromaDB and Neo4j.
- Hybrid rank fusion is implemented and returns neural, symbolic, and overlapping results.
- Automatic CUDA/MPS/CPU selection and CPU worker controls are implemented.
- CUDA embedding inference is verified on the local RTX 3050 with PyTorch `2.12.1+cu126`.
- The existing five-query comparison averages `+0.56` NBR and `+0.10` RDI versus the vector baseline, but has no relevance judgments and is not a final benchmark.
- Retrieval diagnostics explain ranks, citation degree, source mix, and cutoff decisions.
- Structured contradiction scoring and exact verdict parsing are implemented and tested.
- Thirty-seven pytest cases cover retrieval, claim provenance, metrics, logging, contradiction verdicts, citation guards, Neo4j failures, and hypothesis validation.
- Evidence-ranked hypothesis validation is implemented; weak/invalid generations are retained separately for audit.
- Sentence-level claim provenance is implemented; the latest live check accepted 5/5 claims with 9/9 valid passage citations.
- The core prototype implementation is complete; the next milestone is the benchmark and evaluation framework.

Not yet implemented or validated:

- judged query sets and frozen evaluation splits
- standard IR, contradiction, review, and hypothesis metrics
- BM25, graph-only, standard RAG, and rule-based baselines
- scientific NER, embedding, and LLM comparisons
- semantic entailment verification for claim/evidence pairs
- controlled ablations, repeated runs, confidence intervals, and significance tests
- measured corpus scaling and evaluation-focused UI reporting

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the complete implementation inventory, limitations, and ordered future roadmap.

## Project Structure

```text
nesy-graphrag/
|-- app/
|   `-- streamlit_app.py
|-- src/
|   |-- ingestion/
|   |   |-- arxiv_fetcher.py
|   |   |-- semantic_scholar_fetcher.py
|   |   |-- run_ingestion.py
|   |   `-- ner_extractor.py
|   |-- pipeline/
|   |   |-- baseline_harness.py
|   |   |-- contradiction.py
|   |   |-- hypothesis.py
|   |   |-- metrics.py
|   |   |-- orchestrator.py
|   |   |-- prompts.py
|   |   |-- provenance.py
|   |   |-- retrieval.py
|   |   |-- results_logger.py
|   |   |-- review.py
|   |   |-- validator.py
|   |   `-- verdicts.py
|   |-- storage/
|   |   |-- chroma_store.py
|   |   `-- neo4j_store.py
|   `-- utils/
|       |-- compute.py
|       |-- config.py
|       `-- groq_client.py
|-- tests/
|   |-- conftest.py
|   |-- test_core_guards.py
|   |-- test_hypotheses.py
|   |-- test_provenance.py
|   |-- test_retrieval.py
|   `-- test_verdicts.py
|-- PROJECT_STATUS.md
|-- SETUP.md
|-- pytest.ini
|-- requirements.txt
`-- README.md
```

Generated data lives under `data/`, which is ignored by Git.

## Setup

Follow [SETUP.md](SETUP.md) for the verified Windows installation, local Neo4j configuration, CUDA/CPU choices, environment variables, from-scratch data build, validation commands, clean rebuild procedure, and troubleshooting.

## Requirements

- Python 3.11, 64-bit
- A dedicated local Neo4j database
- Groq API key
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
NEO4J_ALLOW_RESET=true

GROQ_API_KEY=your_groq_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key

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
```

Useful defaults are defined in `src/utils/config.py`.

`NEO4J_ALLOW_RESET=true` is required only for a dedicated project database because graph loading clears the configured database.

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

These are prototype diagnostics, not established scientific benchmarks. TS uses valid passage citations and claim coverage, NBR measures graph participation rather than relevance, and HNS measures normalized graph-path distance rather than expert-assessed novelty. See `PROJECT_STATUS.md` for limitations.

## Scope Boundaries

- The current corpus is 8,850 cleaned Semantic Scholar records, primarily abstracts and metadata. It is not yet the million-scale full-text corpus proposed in the Phase 1 report.
- The project uses pretrained SPECTER embeddings and the configurable Groq model `openai/gpt-oss-120b`; it does not train or fine-tune a local reasoning model.
- Review claims are traceable to deterministic sentence IDs from verified abstracts. Passage existence is enforced; semantic entailment quality is not yet benchmarked.
- PDF ingestion and full-text section provenance remain intentionally deferred for this phase.

## Development Notes

- Keep credentials in `.env`; do not commit them.
- Keep generated datasets and ChromaDB files under `data/`.
- Rebuild the S2 Chroma collection before final evaluation if its vector count does not match the current cleaned S2 dataset.
- Resource usage changes by stage: embedding can use the GPU, NER uses GPU when supported or parallel CPU otherwise, and API/Neo4j stages can be network or database bound.
- Pytest unit tests use small in-memory driver doubles only to isolate failure/guard behavior; end-to-end checks use the real local Neo4j graph and Chroma index.
- Use `PROJECT_STATUS.md` for planning updates instead of creating new phase/status files.
