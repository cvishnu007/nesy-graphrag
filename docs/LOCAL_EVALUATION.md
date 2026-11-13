# Local demo and evaluation runbook

> Read [AGENT_HANDOFF.md](../AGENT_HANDOFF.md) first for the current scope, local store snapshot, incomplete checkpoints, and data-safety rules. This document describes evaluation/reproduction procedures. Historical macOS references are examples, not a requirement for the current Windows environment.

**Current checkpoint:** retrieval grading remains paused at 778/1,145 labels. The current local data and stores have been reconciled to **52,822 raw / 47,619 clean / 47,619 NER / 47,619 Chroma / 47,619 Neo4j papers** with 22,370 citation edges. Ollama and Streamlit are stopped unless the operator starts them. The paused pool in `results/retrieval/expanded_20260913/` was frozen on a different 47,627-paper corpus. Resuming `grade`/`evaluate` is safe because they use only saved candidates and rankings, but `matched_rag` runs live on the current 47,619-paper stores, so label each result with its corpus. These commands are resumable/reproducible procedures, not a claim that a live demo is already running.

Historical cleanup and macOS-specific notes remain in [Chinmay.md](../Chinmay.md). They do not override the current snapshot in [AGENT_HANDOFF.md](../AGENT_HANDOFF.md). Abstracts and generated stores remain ignored by Git.

Run commands from the repository root. Activate the existing environment with `source venv/bin/activate` on macOS/Linux, or `.\venv\Scripts\Activate.ps1` on Windows, then use `python` below. Install `requirements.txt` and `en_core_web_sm` for a fresh environment. Do not recreate an existing environment.

## Local generation

Install [Ollama](https://docs.ollama.com/) and run:

```sh
ollama serve
# In another terminal:
ollama pull qwen2.5:3b-instruct
```

After reinstalling on macOS, `brew services start ollama` can manage the service; a second `ollama serve` is unnecessary. The verified installation is Ollama 0.33.3. Set these values in the ignored `.env`:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:3b-instruct
OLLAMA_TIMEOUT_SEC=300
OLLAMA_CONTEXT_LENGTH=8192
OLLAMA_KEEP_ALIVE=2m
LLM_MAX_RETRIES=4
```

Review, contradiction, hypothesis, Streamlit, and general evaluation collectors use the same completion interface. Ollama needs no Groq key. Optional `OLLAMA_FALLBACK_MODEL` must already be installed and is used only when the primary model is missing. Connection failures and timeouts use bounded retries; the application never downloads a model or changes provider automatically.

For hosted generation, set `LLM_PROVIDER=groq`, a real `GROQ_API_KEY`, `LLM_MODEL`, and `LLM_MODEL_FALLBACK`. The explicitly Groq-specific model-comparison and historical annotation experiments remain provider-specific.

## Dedicated Neo4j and store rebuild

Use a dedicated Neo4j database and put its real connection credentials in `.env`. The September 13 Neo4j 2026.07.1 instance stored its configuration/data/logs under `data/neo4j-local/`. Those files were removed during cleanup. If recreating that layout, its start command is:

```sh
NEO4J_CONF="$PWD/data/neo4j-local/conf" neo4j start
```

The removed configuration used loopback Bolt 7687 and HTTP 7474, a 512 MB maximum heap, and a 128 MB page cache. Its generated password was removed when the original `.env` was restored. The pre-session backup was removed after byte-exact restoration of the original environment and JSON files.

On a new machine, create a dedicated database using Neo4j Desktop or the server setup instructions in [SETUP.md](../SETUP.md). Do not copy this Mac's absolute paths into another installation.

Ingestion and indexing resume from checkpoints/existing IDs:

```sh
python -m src.ingestion.run_ingestion
SPACY_DEVICE=cpu SPACY_N_PROCESS=2 NER_CHECKPOINT_SIZE=1000 python -m src.ingestion.ner_extractor
EMBEDDING_BATCH_SIZE=8 python -m src.storage.chroma_store
NEO4J_ALLOW_RESET=true python -m src.storage.neo4j_store
```

The last command rebuilds the dedicated graph. Reset uses 1,000-node transactions to avoid a whole-graph transaction exceeding memory; this syntax requires Neo4j 5.23 or later ([Neo4j documentation](https://neo4j.com/docs/cypher-manual/current/subqueries/subqueries-in-transactions/)). Keep `NEO4J_ALLOW_RESET=false` in `.env` and enable it for that command only. On PowerShell use `$env:NAME="value"` before a command and remove the override afterward. On an 8 GB Mac, CPU NER avoids competing with MPS embedding inference; run generation after indexing has finished. `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` can be used once SPECTER is cached.

```sh
python -m src.evaluation.store_audit --output results/local/store_audit_new.json
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
python -m src.evaluation.live_completion_checks --external-authorized --output-dir results/reasoning/live_local_new
```

Live checks are engineering smoke tests. Inspect accepted claim/verdict/hypothesis counts and saved validity details; a completed call alone does not establish scientific correctness.

## Offline packaged reasoning reproduction

This command runs without Neo4j, Chroma, Ollama, or Groq:

```sh
python -m src.evaluation.packaged_reproduction --output-dir results/reasoning/package_reproduction_new
```

It resolves inputs directly under `teammate2/`, verifies the frozen benchmark hashes (allowing CRLF/LF differences), validates all 392 annotation assignments for 313 unique items, and scores development/test predictions for 62 contradiction, 151 support, and 100 hypothesis items. Historical package-manifest drift is reported separately; the original manifest is retained. `teammate2/reproduce.ps1` wraps this command with a fresh timestamped output directory.

Default reasoning benchmark paths now resolve to the packaged files regardless of working directory. Other data/result paths still follow configuration. Historical package labels are AI-generated references, not human ground truth.

## Expanded retrieval and matched-context RAG

Use a fresh directory for each corpus/protocol version. `prepare` requires exact corpus/Chroma/Neo4j paper-ID parity and freezes all five methods' rankings before grading:

```sh
python -m src.evaluation.refresh_retrieval prepare --directory results/retrieval/expanded_new
python -m src.evaluation.refresh_retrieval grade --directory results/retrieval/expanded_new --batch-size 4
python -m src.evaluation.refresh_retrieval evaluate --directory results/retrieval/expanded_new
```

Methods: vector-only, graph-only, existing evaluation hybrid (16:1), BM25 (k1=1.5, b=0.75; title plus abstract), and production graph-filtered fusion. No parameter is selected using new test labels. The original 6 development / 14 test queries are unchanged. Pool depth is 20; recall is relative to the judged pool, not exhaustively known corpus relevance. BM25 uses title plus abstract whereas historical SPECTER indexing uses abstracts, an explicit difference in the baseline representations.

The local model grades method-hidden title/abstract evidence with checkpointed raw responses and [Ollama JSON-schema output constraints](https://docs.ollama.com/capabilities/structured-outputs). Development-only formatting trials are retained separately and their labels discarded before test-query grading. The final schema returns ordered integer grades mapped to saved batch IDs, avoiding redundant ID/prose generation; AI rationales are not requested. `judgments_for_review.csv` contains AI suggestions beside the blank human-review fields. These are provisional AI-reference scores requiring human review. Frozen hashes reject changed candidates, ranks, or labels. Evaluation scores captured rankings; its cache-access timings are **not live retrieval latency**.

```sh
python -m src.evaluation.matched_rag --split test --output-dir results/reasoning/matched_rag_new
```

Both arms use the same local model, prompt, temperature 0, top 5 papers, 6,000 formatted Unicode characters of evidence, and 600 output tokens, with zero repair attempts. The budget is characters, not tokenizer tokens. Each arm validates IDs against the clean corpus and retains whole evidence sentences. Outputs preserve prompts' hashes, passages, raw generations, and structural provenance. This comparison measures traceability and generation behavior; it does not yet have independent answer-quality labels. `--query-limit` produces an explicitly marked smoke sample.

## Reasoning coverage and partial support

```sh
python -m src.evaluation.support_calibration --output-dir results/reasoning/support_calibration_new
python -m src.evaluation.scifact_evaluation --output-dir results/reasoning/scifact_nli_new
```

Calibration chooses a partial-entailment threshold using saved development probabilities only, persists the choice, then opens the disjoint test file. The historical AI-reference test improved partial-support F1 from 0 to 0.125, but false acceptance increased from 0 to 0.03125. This experimental threshold has **not** been adopted in production. Because previously observed test weakness motivated this experiment, a fresh untouched test set is needed for a confirmatory improvement claim.

The separate [SciFact](https://github.com/allenai/scifact) benchmark imports published expert annotations, keeps complete evidence rationales together, and excludes duplicate content across partitions. Official train becomes local development (955 items); official dev becomes local test (338 items, including 122 contradicted). It evaluates biomedical claim/evidence classification; it does not validate graph discovery or paper-to-paper contradiction and contains no partial/unsupported gold classes. The old frozen graph-pair benchmark still has only one positive test contradiction.

For the SciFact inference command, first cache the pinned model once:

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('cross-encoder/nli-deberta-v3-small', revision='fa2804872c3b4bd748f38c0185cc85775361e735', allow_patterns=['*.json','*.txt','*.model','*.safetensors','pytorch_model.bin'])"
```

To score the committed predictions without any model download or inference:

```sh
python -m src.evaluation.scifact_evaluation --predictions results/reasoning/scifact_nli_20260913/predictions.jsonl --output-dir results/reasoning/scifact_rescore_new
```

The fixed floor is 0.20; there was no SciFact test tuning. Macro F1 averages the two gold classes; all four prediction classes remain in the confusion matrix. Source claims/labels are CC BY 4.0; abstracts are ODC-By 1.0, as documented in the [source license](https://github.com/allenai/scifact/blob/master/LICENSE.md). To rebuild the imported benchmark from the official archive, extract it to `data/scifact/` and run `python -m src.evaluation.scifact_benchmark --output evaluation/benchmarks/scifact_support_v2.json`.
