# NeSy-GraphRAG

NeSy-GraphRAG is a research-assistant prototype for automated literature review and hypothesis generation. It combines neural retrieval with symbolic graph validation so generated reviews are grounded in papers that exist in the project knowledge graph.

The current focus is the GraphRAG pipeline, citation validation, contradiction detection, hypothesis generation, and evaluation against a vector-only baseline. PDF ingestion is not part of the current scope.

## What It Does

- Retrieves relevant scientific papers with SPECTER embeddings and ChromaDB.
- Expands results through a Neo4j citation/concept graph.
- Validates paper IDs against Neo4j before sending context to the LLM.
- Generates literature reviews through Groq-hosted Llama models.
- Detects possible contradiction candidates across related papers.
- Generates hypothesis candidates from structural holes in the graph.
- Computes evaluation metrics for trustworthiness, graph contribution, temporal diversity, reasoning depth, and hypothesis novelty.
- Provides a Streamlit UI for interactive use.

## Current Status

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the single source of truth on what is done, what is blocked, and what needs to be implemented next.

Short version:

- ArXiv and Semantic Scholar ingestion are implemented.
- Local ArXiv and S2 data files exist.
- ChromaDB indexing is implemented and local collections exist.
- Neo4j graph loading code is implemented.
- Literature review, contradiction, hypothesis, metrics, and baseline comparison modules exist.
- Static Python compilation passes.
- The next milestone is a clean end-to-end S2 run with Neo4j reachable and the `s2_papers` Chroma collection rebuilt.

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
|   |   |-- retrieval.py
|   |   |-- review.py
|   |   |-- validator.py
|   |   `-- verdicts.py
|   |-- storage/
|   |   |-- chroma_store.py
|   |   `-- neo4j_store.py
|   `-- utils/
|       |-- config.py
|       `-- groq_client.py
|-- PROJECT_STATUS.md
|-- requirements.txt
`-- README.md
```

Generated data lives under `data/`, which is ignored by Git.

## Requirements

- Python 3.10+
- Neo4j, either local or AuraDB
- Groq API key
- Semantic Scholar API key for S2 ingestion

Install Python dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Configuration

Create a local `.env` file in the project root:

```env
DATA_SOURCE=s2

NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

GROQ_API_KEY=your_groq_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key

EMBEDDING_MODEL=allenai-specter
LLM_MODEL=llama-3.3-70b-versatile
LLM_MODEL_FALLBACK=llama-3.1-8b-instant
```

Useful defaults are defined in `src/utils/config.py`.

## Pipeline

Run the data and graph setup in this order:

```bash
python -m src.ingestion.run_ingestion
python -m src.ingestion.ner_extractor
python -m src.storage.chroma_store
python -m src.storage.neo4j_store
```

Run the full pipeline smoke test:

```bash
python -m src.pipeline.orchestrator
```

Run the baseline comparison:

```bash
python -m src.pipeline.baseline_harness
```

## App

Start the Streamlit UI:

```bash
streamlit run app/streamlit_app.py
```

The app supports:

- Literature Review
- Contradiction Detection
- Hypothesis Generation

## Metrics

Implemented in `src/pipeline/metrics.py`:

- `TS`: Trustworthiness Score
- `NBR`: NeSy Boost Ratio
- `ATD`: Answer Temporal Diversity
- `RDI`: Reasoning Depth Index
- `HNS`: Hypothesis Novelty Score

## Development Notes

- Keep credentials in `.env`; do not commit them.
- Keep generated datasets and ChromaDB files under `data/`.
- Rebuild the S2 Chroma collection before final evaluation if its vector count does not match the current cleaned S2 dataset.
- Use `PROJECT_STATUS.md` for planning updates instead of creating new phase/status files.
