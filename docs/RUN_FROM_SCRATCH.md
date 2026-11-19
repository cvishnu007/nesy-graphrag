# Running NeSy-GraphRAG From Scratch

This guide goes from an empty Windows machine to the running app and the reproduced paper results. Commands are for PowerShell, run from the repository root.

There are three ways to use the project. Pick the one you need:

| Goal | Sections | Time |
|---|---|---|
| **A. Check the paper's numbers** (no databases, no models) | 1–4, then 13 | ~30 min |
| **B. Run the app on the existing data** (someone gives you `data/` and the Neo4j database) | 1–7, then 12 | ~1 h |
| **C. Rebuild everything** (download the corpus and build every store yourself) | 1–12 | 1–3 days, mostly unattended |

---

## 1. Prerequisites

Install these first:

| Tool | Version used | Notes |
|---|---|---|
| Git | any recent | |
| Python | 3.11, 64-bit | Python 3.13 also works for CPU-only evaluation |
| Neo4j | 2026.07.1 (Desktop or Community Server) | Needed for B and C. Use a **dedicated** database: rebuilding the graph clears it |
| Ollama | any recent | Needed for B and C. Runs the local language model |
| NVIDIA driver | supports CUDA 12.6 | Optional. Without a GPU everything runs on CPU, more slowly |

Hardware used for every result: Windows 11, 16 GB RAM, NVIDIA RTX 3050 Laptop GPU (4 GB). With 16 GB RAM, close heavy programs while the 7B model runs; it needs about 3–4 GB free.

API keys (free), needed only to rebuild the data (C):
- **Semantic Scholar:** <https://www.semanticscholar.org/product/api>
- **OpenAlex:** <https://openalex.org/> (used for extra citation links)

## 2. Clone

```powershell
git clone --branch master https://github.com/cvishnu007/nesy-graphrag.git
Set-Location .\nesy-graphrag
```

## 3. Python environment

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

With an NVIDIA GPU, install the CUDA build of PyTorch first (about 2.6 GB):

```powershell
.\venv\Scripts\python.exe -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
```

Then the project dependencies and the spaCy model:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m spacy download en_core_web_sm
```

Check the GPU (prints `True` and the GPU name, or `False` on CPU):

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

All commands below call `.\venv\Scripts\python.exe` directly, so activating the environment is optional.

## 4. Run the tests

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

All tests should pass. They use no databases or models. For goal A, go to section 13 now.

## 5. Configure `.env`

```powershell
Copy-Item .env.example .env
```

Open `.env` and set at least:

```dotenv
NEO4J_PASSWORD=your_neo4j_password
SEMANTIC_SCHOLAR_API_KEY=your_key      # only for rebuilding data (C)
OPENALEX_API_KEY=your_key              # only for rebuilding data (C)
```

`.env.example` already contains the configuration used in the paper (BGE retrieval, BM25 fusion, graph expansion, scientific entities, citation-grounded contradictions, the method–task hypothesis ranker, local `qwen2.5:7b-instruct`).

Safety rules:
- Never commit `.env`. It is ignored by Git.
- Keep `NEO4J_ALLOW_RESET=false`. Section 9 turns it on for one command only, because that command clears the configured Neo4j database.

## 6. Start Neo4j and Ollama

**Neo4j:** create a local database (Neo4j Desktop: *Add → Local DBMS*), set the password you put in `.env`, and start it. Check the connection:

```powershell
.\venv\Scripts\python.exe -c "from src.storage.neo4j_store import get_driver; d=get_driver(); d.verify_connectivity(); print('Neo4j OK'); d.close()"
```

**Ollama:** install it, then download the model (about 4.7 GB):

```powershell
ollama pull qwen2.5:7b-instruct
ollama list
```

Ollama runs as a background service after installation. If `ollama list` fails, start it with `ollama serve` in a separate window.

## 7. Get the data

For goal B, copy the shared `data/` folder into the repository root and restore the shared Neo4j database dump (Neo4j Desktop: *… → Load from file*). Then skip to section 12.

For goal C, continue with sections 8–11. Each stage resumes if interrupted. Do not delete its output to restart it, and do not run two stages at the same time.

## 8. Build the corpus and the vector index

**8.1 Download the papers** (network-bound, several hours; 11 CS topics, up to 5,000 papers each, 2020–2025):

```powershell
.\venv\Scripts\python.exe -m src.ingestion.run_ingestion
```

Output: `data/s2_raw.json` and `data/s2_clean.json`. The paper's corpus has 52,822 raw and 47,619 clean papers. A new download can differ slightly because Semantic Scholar changes over time.

**8.2 spaCy entities** (used by the base graph):

```powershell
.\venv\Scripts\python.exe -m src.ingestion.ner_extractor
```

**8.3 BGE vector index** (the retrieval index used by the app; GPU recommended):

```powershell
$env:CHROMA_COLLECTION="s2_papers_bge"; $env:EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"; $env:EMBEDDING_TEXT_MODE="title_abstract_plain"
.\venv\Scripts\python.exe -m src.storage.chroma_store
```

**8.4 SPECTER vector index.** It is only used to build the similarity edges in 9.2:

```powershell
$env:CHROMA_COLLECTION="s2_papers"; $env:EMBEDDING_MODEL="allenai-specter"; $env:EMBEDDING_TEXT_MODE="abstract"
.\venv\Scripts\python.exe -m src.storage.chroma_store
Remove-Item Env:CHROMA_COLLECTION, Env:EMBEDDING_MODEL, Env:EMBEDDING_TEXT_MODE
```

Check: each collection should hold as many vectors as `data/s2_clean.json` has papers.

## 9. Build the Neo4j graph

**9.1 Base graph:** papers, authors, spaCy concepts and citations. This **clears the configured database**:

```powershell
$env:NEO4J_ALLOW_RESET = "true"
.\venv\Scripts\python.exe -m src.storage.neo4j_store
Remove-Item Env:NEO4J_ALLOW_RESET
```

**9.2 Additive enrichment** (nothing is deleted): generic-concept flags and `SIMILAR_TO` edges:

```powershell
.\venv\Scripts\python.exe -m src.storage.graph_enrichment concepts --write
.\venv\Scripts\python.exe -m src.storage.graph_enrichment similarity --collection s2_papers --write
```

## 10. Scientific entities (SciERC)

Download the SciERC dataset, fine-tune SciBERT on it (about 30 minutes on the RTX 3050), tag every abstract, and load the entities:

```powershell
$env:HF_HUB_DISABLE_XET="1"
.\venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('zj88zj/SCIERC', repo_type='dataset', local_dir='data/external/zj88zj__SCIERC')"
.\venv\Scripts\python.exe -m src.ingestion.train_scierc_ner
.\venv\Scripts\python.exe -m src.ingestion.scientific_ner extract --model models/scibert-scierc-ner
.\venv\Scripts\python.exe -m src.storage.graph_enrichment scientific-entities --write
```

The model is saved to `models/scibert-scierc-ner/`, which is not committed. It should reach about 0.64 typed F1 on the SciERC test set.

## 11. Citation densification and the contradiction index

**11.1 Fetch reference lists** (network-bound, several hours), build the plan, and write the extra citation layers. All writes are additive:

```powershell
.\venv\Scripts\python.exe -m src.ingestion.citation_densify fetch-s2
.\venv\Scripts\python.exe -m src.ingestion.citation_densify fetch-openalex
.\venv\Scripts\python.exe -m src.ingestion.citation_densify build
.\venv\Scripts\python.exe -m src.ingestion.citation_densify fetch-external
.\venv\Scripts\python.exe -m src.ingestion.citation_densify write --write
.\venv\Scripts\python.exe -m src.ingestion.citation_densify stats
```

Expected: about 133,500 `CITES_ADDED` and 411,400 `COUPLED_WITH` edges, and isolated papers falling from about 71% to about 2%.

**11.2 NLI model for contradiction screening:**

```powershell
.\venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('cross-encoder/nli-deberta-v3-small')"
```

**11.3 Contradiction index.** It pairs critical findings with the papers they cite and runs the 7B verdict on the top 150, which takes about an hour. A copy of the paper's index is already committed in `results/reasoning/contradiction_index_20261114/`, and the app uses that copy by default. Rebuild only after changing the corpus:

```powershell
.\venv\Scripts\python.exe -m src.evaluation.contradiction_index --output-dir results\reasoning\contradiction_index_new
```

To use a rebuilt index, set `CONTRADICTION_INDEX_FILE=results/reasoning/contradiction_index_new/index.jsonl` in `.env`.

## 12. Run the app

With Neo4j and Ollama running:

```powershell
.\venv\Scripts\python.exe -m streamlit run app\streamlit_app.py --server.address localhost
```

Open <http://localhost:8501>. The sidebar shows the paper count, vector count and model in use. The four modes:

| Mode | What you get |
|---|---|
| 📚 Literature Review | A cited summary; every claim links to a sentence from a retrieved abstract, and invalid citations are blocked and listed |
| ⚡ Contradiction Detection | Paper pairs where one disputes a paper it cites, with the two conflicting sentences |
| 💡 Hypothesis Generation | "Method → task" ideas not yet combined in the corpus, with supporting evidence. The first query takes about a minute |
| 🔗 Related Work | An interactive citation map plus a list of papers to cite, each with the citation records behind it |

Example queries for each mode: [docs/DEMO_QUERIES.md](DEMO_QUERIES.md). If the review shows a red "did not return an answer" error, the PC is short of RAM for the model: close other programs and run it again.

## 13. Reproduce the paper's results

**13.1 Headline numbers from saved artifacts** (no databases or models; seconds):

```powershell
.\venv\Scripts\python.exe -m src.evaluation.reproduce
```

It recomputes 23 headline values (judge agreement, retrieval NDCG, human agreement and ratings, the LightRAG baseline, citation recommendation on our corpus, DBLP and PubMed, and the time-sliced discovery benchmark) and prints `23/23 reproduced`. Two checks need the gitignored key `evaluation/human_review/relevance_subset50_key_DO_NOT_SHARE_WITH_ANNOTATORS.json`; ask the project owner for it.

**13.2 Rerun experiments.** These need the stores; LLM steps need Ollama. Each writes to its own results folder. Run them as `.\venv\Scripts\python.exe -m ...`; each module's docstring (or `--help`) explains its stages:

| Paper section (RESULTS.md) | Command |
|---|---|
| §2b judge validation | `python -m src.evaluation.judge_agreement --help` |
| §2d 44-query retrieval benchmark | `python -m src.evaluation.stratified_benchmark score` |
| §2f citation recommendation (our corpus) | `python -m src.evaluation.citation_recommendation --help` |
| §2g LightRAG-style baseline | `python -m src.evaluation.lightrag_baseline build`, then `keywords`, `rank`, `grade`, `score` |
| §2h DBLP / PubMed | download two files (below), then `python -m src.evaluation.citeomatic_benchmark --dataset dblp --split test [--encoder bge\|specter2_base\|scincl]` |
| §4c–4d time-sliced discovery | `python -m src.evaluation.temporal_hypothesis` |
| Human ratings | `python -m src.evaluation.human_ratings {hypothesis,contradiction,hypothesis_round2,contradiction_round2} --sheets ...` |

DBLP and PubMed files (304 MB in total; only these two are needed, not the 47 GB archive):

```powershell
$B="https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/citeomatic-2018-02-12/comparison"
New-Item -ItemType Directory -Force data\external\citeomatic\dblp, data\external\citeomatic\pubmed | Out-Null
curl.exe -L -o data\external\citeomatic\dblp\corpus.json "$B/dblp/corpus.json"
curl.exe -L -o data\external\citeomatic\pubmed\corpus.json "$B/pubmed/corpus.json"
```

Run grading jobs one at a time, and leave the PC on: the LLM judge steps take hours. Never edit a frozen results folder; every rerun writes to a new one.

## 14. Troubleshooting

| Problem | Fix |
|---|---|
| `No module named ...` | Use `.\venv\Scripts\python.exe`, not a global `python` |
| `Can't find model 'en_core_web_sm'` | `.\venv\Scripts\python.exe -m spacy download en_core_web_sm` |
| CUDA shows `False` | Reinstall the CUDA PyTorch build from section 3; check `nvidia-smi` |
| Neo4j connection fails | Start the database; check the URI (`neo4j://127.0.0.1:7687`) and password in `.env` |
| `Ollama server failed after 4 attempts` | Not enough free RAM: Ollama's log says "model request too large for system". Close other programs; with 16 GB RAM, don't run two model jobs at once |
| Hugging Face download hangs | Set `$env:HF_HUB_DISABLE_XET="1"` and retry |
| Semantic Scholar `429` | Normal rate limiting; the client retries with back-off. Run only one ingestion process |
| Chroma count ≠ paper count | The collection is stale. Rebuild that collection (8.3 or 8.4) |
| Windows console `UnicodeEncodeError` | Set `$env:PYTHONIOENCODING="utf-8"` before the command |

## Where to read next

- [README.md](../README.md): project overview and headline results
- [docs/RESULTS.md](RESULTS.md): every result table with its source files
- [docs/PAPER_DRAFT.md](PAPER_DRAFT.md): the paper draft
- [AGENT_HANDOFF.md](../AGENT_HANDOFF.md): detailed state, decisions and safety rules for continuing development
