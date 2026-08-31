# NeSy-GraphRAG Setup

This guide reproduces the verified local project setup from a fresh clone. The primary instructions target Windows PowerShell, Python 3.11, a local Neo4j database, Semantic Scholar data, and an optional NVIDIA GPU.

PDF ingestion is not part of this setup. The current pipeline uses paper metadata and abstracts.

## Tested Stack

- Windows 10/11
- Python 3.11.9, 64-bit for the primary CUDA environment
- Python 3.13.14, 64-bit also verified for CPU-only evaluation
- Local Neo4j `2026.07.1` using Bolt at `neo4j://127.0.0.1:7687`
- NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM
- PyTorch `2.12.1+cu126`
- Python dependencies pinned in `requirements.txt`
- Semantic Scholar corpus configuration: 11 CSE topic queries, up to 5,000 records per query, 2020-2025, Computer Science
- SPECTER embeddings: `allenai-specter`
- Groq primary model: `openai/gpt-oss-120b`
- Groq fallback model: `llama-3.1-8b-instant`
- Retrieval benchmark: 20 queries (6 development, 14 test), version `0.2-draft`
- Current benchmark judgments: machine-assisted and pending human review

Other hardware can run the project. `EMBEDDING_DEVICE=auto` selects CUDA, then Apple MPS, then CPU.

Python 3.13.14 has also been used successfully for CPU-only retrieval evaluation. The instructions below retain Python 3.11 because that is the verified CUDA setup used for the primary local machine.

## 1. Prerequisites

Install these before cloning:

1. Git
2. Python 3.11, 64-bit
3. Neo4j Desktop, Neo4j Community Server, or another local Neo4j installation
4. A Groq API key
5. A Semantic Scholar API key for reliable multi-topic ingestion
6. An NVIDIA driver that supports the CUDA 12.6 PyTorch wheel, if using NVIDIA acceleration

The full CUDA Toolkit is not required by the PyTorch wheel, but a compatible NVIDIA driver is required.

Confirm the basic tools:

```powershell
git --version
py -3.11 --version
```

## 2. Clone The Repository

```powershell
git clone --branch master --single-branch https://github.com/cvishnu007/nesy-graphrag.git
Set-Location .\nesy-graphrag
git status --short --branch
```

The final command should show `master` with a clean worktree.

### Before Making Any Changes

Always synchronize the local `master` branch before editing code, documentation, configuration, or tests:

```powershell
git switch master
git status --short --branch
git pull --ff-only origin master
```

Do not begin new work until `master` is up to date and `git status` shows a clean worktree. Resolve or preserve any existing local changes before pulling. After the pull succeeds, create or switch to the branch where the new work will be committed:

```powershell
git switch -c your-branch-name
```

For every later change, repeat the synchronization step against `master` first. Using `--ff-only` prevents Git from silently creating an unintended merge commit during the pull.

## 3. Create Or Reuse The Virtual Environment

For a fresh clone:

```powershell
py -3.11 -m venv venv
```

If `venv` already exists, do not recreate it. Verify its interpreter first:

```powershell
.\venv\Scripts\python.exe --version
.\venv\Scripts\python.exe -m pip --version
```

All commands below use the explicit virtual-environment interpreter. Activating the environment is optional.

Upgrade packaging tools:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

## 4. Install PyTorch And Project Dependencies

### NVIDIA GPU Setup

Install the tested CUDA 12.6 build first:

```powershell
.\venv\Scripts\python.exe -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
```

Then install the remaining pinned dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verify CUDA:

```powershell
.\venv\Scripts\python.exe -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The verified machine reports `2.12.1+cu126`, `True`, and `NVIDIA GeForce RTX 3050 Laptop GPU`.

### CPU-Only Setup

Skip the CUDA-specific command and run:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set `EMBEDDING_DEVICE=cpu` in `.env`, or leave it as `auto` to fall back automatically.

### Install The spaCy Model

```powershell
.\venv\Scripts\python.exe -m spacy download en_core_web_sm
```

Check dependency consistency:

```powershell
.\venv\Scripts\python.exe -m pip check
```

## 5. Configure Neo4j

Create and start a dedicated local Neo4j database. The graph build command executes `MATCH (n) DETACH DELETE n`, so do not point this project at a database containing unrelated data.

Verified local connection settings:

- URI: `neo4j://127.0.0.1:7687`
- Username: `neo4j`
- Password: the password selected when creating the local database

The code uses Neo4j constraints and Cypher syntax supported by modern Neo4j releases. The verified server is `2026.07.1`.

Keep the Neo4j service running during graph loading, retrieval, tests that use live services, and Streamlit use.

## 6. Create The Environment File

Copy the tracked template:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and replace these values:

```env
NEO4J_PASSWORD=your_local_neo4j_password
GROQ_API_KEY=your_groq_api_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
```

The template already matches the verified project configuration:

```env
DATA_SOURCE=s2
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_ALLOW_RESET=true

S2_QUERY=graph neural networks
S2_QUERIES=graph neural networks;artificial intelligence and machine learning;cybersecurity;computer networks;databases;software engineering;cloud computing;natural language processing;computer vision;data science;operating systems
S2_LIMIT=10000
S2_LIMIT_PER_QUERY=5000
S2_INCLUDE_EXISTING=true
S2_CHECKPOINT_FILE=./data/s2_ingestion_checkpoint.json
S2_YEAR=2020-2025
S2_FIELDS_OF_STUDY=Computer Science

CHROMA_COLLECTION=s2_papers
USE_REAL_CITATIONS=true

EMBEDDING_MODEL=allenai-specter
EMBEDDING_DEVICE=auto
EMBEDDING_BATCH_SIZE=16

LLM_MODEL=openai/gpt-oss-120b
LLM_MODEL_FALLBACK=llama-3.1-8b-instant

EVALUATION_START_YEAR=2020
EVALUATION_END_YEAR=2025

EVALUATION_HYBRID_VECTOR_WEIGHT=16.0
EVALUATION_HYBRID_GRAPH_WEIGHT=1.0
GRAPH_ONLY_CANDIDATE_LIMIT=100

GRAPH_HIGH_SEMANTIC_THRESHOLD=0.85
GRAPH_SEMANTIC_FLOOR=0.75
GRAPH_MIN_QUERY_TERM_COVERAGE=0.75
GRAPH_STRONG_CONNECTIONS=10
```

`NEO4J_ALLOW_RESET=true` is required for the graph build because it intentionally clears the configured database. Set it only for a dedicated project database.

Do not commit `.env`. It is ignored by Git.

Verify non-secret configuration values:

```powershell
.\venv\Scripts\python.exe -c "from src.utils import config; print(config.DATA_SOURCE, config.NEO4J_URI, config.CHROMA_COLLECTION, config.EMBEDDING_MODEL, config.LLM_MODEL)"
```

## 7. Verify Neo4j Connectivity

Start the Neo4j service, then run:

```powershell
.\venv\Scripts\python.exe -c "from src.storage.neo4j_store import get_driver; d=get_driver(); print('Neo4j connectivity OK'); d.close()"
```

If this fails, fix Neo4j before beginning ingestion. Confirm the service is running, the password is correct, and port `7687` is available.

## 8. Build The Project Data From Scratch

Run each stage from the repository root and wait for it to finish before starting the next stage.

### Stage 1: Semantic Scholar Ingestion

```powershell
.\venv\Scripts\python.exe -m src.ingestion.run_ingestion
```

Expected outputs:

- `data/s2_raw.json`
- `data/s2_clean.json`

`S2_QUERIES` is semicolon-separated. The broad CSE configuration keeps the existing raw corpus, fetches up to 5,000 papers for each configured topic, deduplicates globally by Semantic Scholar paper ID, and saves both the merged raw file and a completed-topic checkpoint after every topic. Rerunning the command resumes completed topics instead of replacing prior data.

The exact retained count can change as Semantic Scholar data changes. The August 31, 2026 verified broad run produced 52,822 unique raw records and 47,619 clean records. It preserved every ID from the original 10,000 raw and 8,850 clean records.

Semantic Scholar rate limits make this stage network-bound. Do not launch duplicate ingestion processes. If the API returns `429`, allow the built-in retry delay to continue.

### Stage 2: Entity Extraction

```powershell
.\venv\Scripts\python.exe -m src.ingestion.ner_extractor
```

Expected output:

- `data/s2_ner.json`

NER reuses entity lists already present in `data/s2_ner.json` and checkpoints new work every 5,000 papers. On CPU, the code uses all but one logical processor unless `SPACY_N_PROCESS` is set. On supported spaCy GPU installations, `SPACY_DEVICE=auto` can select the GPU and uses one process.

### Stage 3: Chroma Index

```powershell
.\venv\Scripts\python.exe -m src.storage.chroma_store
```

Expected output:

- `data/chromadb/`
- Chroma collection `s2_papers`

The first run downloads `allenai-specter` from Hugging Face unless it is already cached. Keep network access enabled. The index builder resumes by skipping IDs already present in the collection.

Verify the vector count:

```powershell
.\venv\Scripts\python.exe -c "from src.storage.chroma_store import get_collection; print('Chroma vectors:', get_collection().count())"
```

The count should match the cleaned-paper count. The verified broad snapshot reports 47,619.

### Stage 4: Neo4j Graph

Ensure Neo4j is running and `.env` contains `NEO4J_ALLOW_RESET=true`.

```powershell
.\venv\Scripts\python.exe -m src.storage.neo4j_store
```

This stage clears the configured Neo4j database and rebuilds papers, authors, concepts, and citation edges from `data/s2_ner.json`.

Verify graph counts:

```powershell
.\venv\Scripts\python.exe -c "from src.storage.neo4j_store import get_driver; d=get_driver(); s=d.session(); print('Papers:', s.run('MATCH (p:Paper) RETURN count(p) AS c').single()['c']); print('CITES:', s.run('MATCH ()-[r:CITES]->() RETURN count(r) AS c').single()['c']); s.close(); d.close()"
```

The paper count should match Chroma. The verified broad snapshot reports 47,619 papers, 145,957 authors, 195,252 concepts, and 22,370 real `CITES` edges. Real citation edges are created only when both the source and referenced target paper exist in the ingested corpus.

## 9. Run Automated Verification

Run the complete unit suite:

```powershell
.\venv\Scripts\python.exe -m pytest
```

The current repository collects 127 pytest cases, including multi-topic ingestion,
resume safety, production retrieval, evaluation, provenance, contradiction, and
hypothesis coverage.

Compile all Python modules:

```powershell
.\venv\Scripts\python.exe -m compileall -q src app tests
```

Run retrieval diagnostics without calling the LLM:

```powershell
.\venv\Scripts\python.exe -m src.pipeline.retrieval "graph neural networks for node classification" --top-k 10
```

The diagnostic prints both the raw and retained graph-candidate counts. A low or
zero retained count is valid: weak citation neighbours are intentionally removed,
while vector results still provide the requested number of papers.

Run a single live review with Neo4j, Chroma, Hugging Face, and Groq:

```powershell
.\venv\Scripts\python.exe -c "from src.pipeline.orchestrator import graphrag_query; r=graphrag_query('graph neural networks for node classification', mode='review', top_k=5); print(r['provenance']['stats'])"
```

A valid run should report verified papers, at least one accepted claim, valid passage citations, and no fabricated passage IDs.

## 10. Run Retrieval Evaluation

Retrieval evaluation requires the Chroma index and Neo4j graph. It does not call Groq. The tracked benchmark contains 20 frozen queries with 6 development and 14 held-out test queries.

The current `0.2-draft` labels are machine-assisted, predate the broad-CSE rebuild, and require candidate-pool refresh plus human review. They are historical development diagnostics, not final expanded-corpus results.

Run the tracked judged-draft benchmark on the held-out split:

```powershell
.\venv\Scripts\python.exe -m src.evaluation.retrieval_runner `
  --benchmark evaluation/benchmarks/retrieval_queries_judged.json `
  --split test `
  --top-k 20 `
  --output-dir results/retrieval/evaluation `
  --overwrite
```

Compare hybrid and vector NDCG@10 with paired significance analysis:

```powershell
.\venv\Scripts\python.exe -m src.evaluation.significance `
  results/retrieval/evaluation/per_query_metrics.csv `
  --challenger hybrid `
  --reference vector `
  --metric ndcg@10 `
  --output results/retrieval/evaluation/significance.json
```

To revise the benchmark correctly:

1. Review `evaluation/benchmarks/retrieval_judgments_draft.csv` without exposing retrieval-method identity.
2. Correct grades using the documented 0/1/2 relevance scale.
3. Record reviewer and adjudication metadata.
4. Finalize to a new benchmark version rather than silently replacing the draft.
5. Tune only on the six development queries.
6. Run the fourteen test queries once after configuration is frozen.

The current evaluation-only hybrid uses a 16:1 vector-to-graph weight. Production retrieval uses a separate filtered 1:1 flow; do not report the two configurations as the same system.

## 11. Start The Streamlit App

```powershell
.\venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

The app provides:

- literature review with claim-level passage evidence
- contradiction candidate evaluation
- evidence-ranked hypothesis generation
- prototype metrics and graph/store counts

Stop the app with `Ctrl+C`.

## Clean Rebuild Procedure

Use this only when intentionally rebuilding generated state.

1. Stop Streamlit and any running pipeline process.
2. Confirm the current directory is the cloned `nesy-graphrag` repository.
3. Remove generated Chroma and JSON data.
4. Start the dedicated Neo4j database.
5. Run Stages 1-4 again in order.

PowerShell cleanup commands from the repository root:

```powershell
Remove-Item -LiteralPath .\data\chromadb -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath .\data\s2_raw.json -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath .\data\s2_clean.json -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath .\data\s2_ner.json -Force -ErrorAction SilentlyContinue
```

The Neo4j stage clears graph data itself after checking `NEO4J_ALLOW_RESET=true`.

## Troubleshooting

### `No module named ...`

Use the repository interpreter explicitly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Do not use a global `python` or `pip` if it points outside `venv`.

### `Can't find model 'en_core_web_sm'`

```powershell
.\venv\Scripts\python.exe -m spacy download en_core_web_sm
```

### CUDA Is `False`

Check the installed build:

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If the version does not contain `+cu126`, reinstall the CUDA wheel from Step 4. Confirm the NVIDIA driver can see the GPU with `nvidia-smi`.

### Neo4j Connection Failure

- Start the Neo4j service.
- Confirm `NEO4J_URI=neo4j://127.0.0.1:7687`.
- Confirm the username and password.
- Confirm another process is not using port `7687`.
- Use a dedicated database and set `NEO4J_ALLOW_RESET=true` only when rebuilding it.

### Semantic Scholar Rate Limits

- Ensure `SEMANTIC_SCHOLAR_API_KEY` is set.
- Run only one ingestion process.
- Keep `SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC=1.05` or increase it.
- Let built-in retries handle `429` and server errors.

### Hugging Face Download Problems

The first embedding run requires network access. If the model is already cached, set these only for intentionally offline runs:

```env
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Do not enable offline mode before the first successful model download.

### Chroma Count Does Not Match The Clean Dataset

The collection may contain stale data from an older run. Follow the clean rebuild procedure, then rebuild Chroma before rebuilding Neo4j.

### Groq Model Failure

Confirm `GROQ_API_KEY` is valid. The configured primary model is `openai/gpt-oss-120b`; the pipeline falls back to `llama-3.1-8b-instant` when the primary model is unavailable.

### Retrieval Evaluation Has No Results

- Start Neo4j and verify Chroma contains the cleaned-paper count.
- Use `retrieval_queries_judged.json`; the unjudged query file intentionally cannot produce final metrics.
- Keep `top_k` at 20 for comparison with the tracked result.
- Use a new output directory or pass `--overwrite` intentionally.
- Do not interpret draft machine-assisted judgments as human ground truth.

## Security And Data Safety

- Never commit `.env` or API keys.
- Use a dedicated Neo4j database; graph building is destructive by design.
- Generated data and Chroma files belong under `data/` and are ignored by Git.
- Review claims have passage-ID provenance, but semantic correctness still requires evaluation.
- Hypotheses are generated research suggestions, not validated scientific findings.

After setup, read `PROJECT_STATUS.md` for implemented capabilities, known limitations, and the evaluation roadmap.
