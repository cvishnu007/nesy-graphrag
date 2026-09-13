# Chinmay — Local generation and evaluation work log

Updated: September 13, 2026. Branch: `feat/local-evaluation`, based on the latest `origin/phase3` (`880d76e`); local `master` was already current at `ee27d00`.

The user requested all five follow-up workstreams and asked that every change be recorded here. The August 27 handoff is preserved below as history; its Neo4j/Groq blockers no longer describe the current local setup.

## Changes made in this continuation

- Read the repository Markdown handoffs, status files, annotation guidance, reports, and the September 2 Ollama specification/plan. Created an isolated worktree and recorded the execution checklist in `docs/superpowers/plans/2026-09-13-local-evaluation-completion.md`.
- Added `src/utils/llm_client.py`: common text completion interface, local Ollama default, native non-streaming chat API, installed-model check, bounded transport/server retries, clear failures, explicit installed-model fallback, and optional Groq adapter. Preserved Groq reasoning-effort forwarding; Ollama neither imports Groq nor requires its key.
- Migrated review, contradiction, hypothesis, orchestrator, baseline harness, Streamlit, general output collection, support evaluation, manifests, and live checks to the selected provider. Preserved provenance validation, structured-output repair, and task-specific verdict/feasibility gates. Provider-specific historical annotation/model-comparison scripts remain explicit Groq experiments.
- Added provider configuration and updated `.env.example`, including a default-false graph-reset guard. Fixed output-collector `.env` loading before model overrides and added tests for both provider routes.
- Installed Ollama 0.33.3, qwen2.5:3b-instruct, Neo4j 2026.07.1 and Java dependencies on this Mac. Created a dedicated loopback Neo4j instance under ignored `data/neo4j-local/`, with a generated password stored only in ignored `.env`. A real local completion returned `OLLAMA_OK` without a Groq key.
- Backed up original JSON data and `.env` under ignored `data/pre-expansion-20260913/`. Resumed the 11-topic Semantic Scholar fetch after a transient network failure. Finished **52,820 raw / 47,627 clean papers**, retaining all original 10,000 raw / 8,850 clean IDs. The teammate's 47,619-paper snapshot was unavailable and describes a different fetch.
- Finished incremental NER and SPECTER/Chroma indexing: **47,627 entity-bearing papers and 47,627 vectors**. NER used two CPU workers because GPU NER competed with MPS indexing on the 8 GB Mac. Expanded Neo4j rebuild completed with **47,627 papers / 146,157 authors / 196,616 concepts / 22,514 real CITES edges**. A read-only audit verified exact IDs and title/abstract equality across clean, NER, Chroma and Neo4j, with zero missing, extra, duplicate or stale-text records (`results/local/store_audit_20260913.json`).
- Fixed canonical reasoning benchmark defaults to `teammate2/benchmarks/`. Added cross-platform `packaged_reproduction.py` and repaired `teammate2/reproduce.ps1`. Verified frozen benchmark hashes, including line-ending normalization, 313 unique annotation items and 392 assignments, and reproduced both splits with zero task failures. Historical package-manifest drift is reported separately rather than rewriting old evidence.
- Added deterministic BM25 and optional production-filtered fusion to retrieval pooling/scoring. Added `refresh_retrieval.py` to freeze rankings/configuration before AI grading, verify artifact hashes, retain raw/checkpointed method-hidden judgments, export human-review CSV, and score saved development/test ranks without retuning.
- Added `matched_rag.py` for vector RAG versus GraphRAG with the same model, prompt, evidence/output budgets and no repair retries. It stores evidence, raw outputs, hashes and provenance; independent answer-quality labels remain separate work.
- Added development-only partial-support calibration over saved NLI probabilities. Test partial-support F1 increased **0 → 0.125**, macro F1 **0.2855 → 0.3190**, but false acceptance increased **0 → 0.03125**. This experimental threshold was not adopted in production.
- Added a separate, licensed SciFact claim/evidence benchmark and reproducible evaluation CLI. It preserves complete rationale sets and separates official train/local development (955 items) from official dev/local test (338 items, 122 contradicted). The fixed NLI model achieved contradiction F1 **0.6286**, two-gold-class macro F1 **0.5143**, and overall accuracy **0.3580**. It evaluates biomedical entailment, not graph paper-pair detection. The frozen in-domain paper-pair benchmark still has one positive test contradiction.
- Added and extended automated tests for provider selection/retries/fallbacks, pipeline routing, BM25, frozen-pool integrity, matched contexts, packaged paths/hashes, development-only calibration, SciFact conversion and label-blinded evaluation. The current full suite passes **304 tests**; compile, dependency consistency and diff-whitespace checks also pass. One upstream spaCy/Click deprecation warning remains.
- Added `docs/LOCAL_EVALUATION.md` and updated README/setup instructions with local startup, safe rebuilds, portable reproduction and interpretation limits. Tightened live checks so empty or invalid structured generations cannot silently count as successful pipeline evidence.

- Fixed the live expanded-graph rebuild failure: the previous single-transaction `DETACH DELETE` exceeded Neo4j's transaction-memory limit. Rebuilds now delete in 1,000-node transactions and consume the result before reporting success (Neo4j 5.23+ syntax). The dedicated-database reset guard remains required.
- Corrected expanded retrieval protocol capture to read fusion weights from the evaluation configuration module and record graph limits, hop depth and production weights; added a regression check. Matched RAG records the same retrieval settings.

- Added `store_audit.py` and a stale-evidence regression test; the audit checks all titles/abstracts as well as paper-ID parity and NER completion.

## Still running or pending

1. Freeze the expanded five-method retrieval pool, grade provisional AI references, evaluate held-out rankings and paired significance.
2. Run matched RAG and all three local application modes, then verify Streamlit startup.
3. Complete local code review and final test/compile/dependency checks, update this log and project status with the actual results, and retain the changes on the local feature branch.

A separate review agent hit its service usage limit; review continues in the primary session. No code or results have been pushed. Independent human judgment review, stronger in-domain contradiction coverage, and expert hypothesis/answer evaluation remain research limitations rather than completed validation.

## Historical August 27 handoff


Date: August 27, 2026
Branch: `chinmay-phase3-s2-work`
Base branch: `phase3`

## Summary

This branch continues the Phase 3 work on top of `phase3`. The main focus was moving the repo from a prototype-only state toward a real Semantic Scholar backed run, while keeping credentials out of tracked git history.

## Completed

- Added the Semantic Scholar API key only to local ignored `.env`.
- Confirmed the API key is not present in tracked files.
- Ran full Semantic Scholar ingestion with `DATA_SOURCE=s2`.
- Fetched 10,000 raw Semantic Scholar papers.
- Cleaned the dataset down to 8,850 usable papers.
- Generated ignored local data files:
  - `data/s2_raw.json`
  - `data/s2_clean.json`
- Ran NER over the cleaned S2 dataset.
- Generated ignored local file:
  - `data/s2_ner.json`
- Confirmed all 8,850 cleaned papers have extracted entities.
- Built Chroma vector index for the S2 dataset.
- Confirmed Chroma collection `s2_papers` contains 8,850 vectors.
- Created a local Python 3.12 virtual environment in ignored `venv/`.

## Code Changes

- Added placeholder-aware config validation in `src/utils/config.py`.
- Updated `src/ingestion/semantic_scholar_fetcher.py` so placeholder S2 keys are not sent as API keys.
- Updated `src/storage/neo4j_store.py` to fail fast when Neo4j credentials are still placeholders.
- Updated `src/pipeline/orchestrator.py` to fail fast when `GROQ_API_KEY` is still a placeholder.
- Tightened graph expansion in `src/pipeline/retrieval.py`:
  - avoids same-seed self matches during CITES traversal
  - counts distinct seed-paper connections
  - removes unused retrieval code
- Added shared contradiction verdict parsing in `src/pipeline/verdicts.py`.
- Updated `src/pipeline/metrics.py` so RDI only counts explicit `VERDICT: CONTRADICTION`.
- Updated HNS scoring in `src/pipeline/metrics.py` to match the documented shortest-path novelty definition.
- Updated `app/streamlit_app.py` to use the shared verdict parser.
- Updated `.env.example` with Groq fallback model and retry settings.
- Updated `requirements.txt` with compatibility fixes:
  - `numpy<2`
  - `scipy<1.18`
  - `click`
- Updated `CHANGES.md` with the actual Phase 3 local run status.

## Verification Done

- Ran Python compile check:
  - `venv/bin/python -m compileall src app run_test.py diff_results.py`
- Ran dependency consistency check:
  - `venv/bin/pip check`
- Ran import checks for all project runtime dependencies.
- Ran RDI/verdict parser sanity checks.
- Verified Chroma collection count:
  - `s2_papers = 8850`
- Ran secret scan excluding ignored `.env` and generated `data/`.
- Confirmed Neo4j now fails clearly when credentials are placeholders.

## Blocked

The remaining end-to-end Phase 3 steps need real local credentials:

- Neo4j load is blocked because `NEO4J_URI` and `NEO4J_PASSWORD` are still placeholders in `.env`.
- Full LLM/orchestrator run is blocked because `GROQ_API_KEY` is still a placeholder in `.env`.
- Before/after fixture refresh is blocked until Neo4j and Groq are both configured.

## What To Do Next

1. Rotate the Semantic Scholar API key because it was pasted in chat.
2. Update local `.env` with the rotated S2 key.
3. Add real Neo4j Aura credentials in local `.env`:
   - `NEO4J_URI`
   - `NEO4J_USERNAME`
   - `NEO4J_PASSWORD`
4. Load S2 NER data into Neo4j:
   - `venv/bin/python -u -m src.storage.neo4j_store`
5. Check the Neo4j graph load result:
   - number of `Paper` nodes should be close to 8,850
   - real `CITES` edges should be created from Semantic Scholar references
6. Add a real Groq API key in local `.env`:
   - `GROQ_API_KEY`
7. Run the full orchestrator smoke test:
   - `venv/bin/python -u -m src.pipeline.orchestrator`
8. Refresh fixtures after a successful live run:
   - `venv/bin/python run_test.py after`
9. Compare before/after outputs:
   - `venv/bin/python diff_results.py`
10. Run the baseline comparison harness for evaluation:
   - `venv/bin/python -m src.pipeline.baseline_harness`

Note: `src.storage.neo4j_store` clears existing Neo4j data before loading the current dataset, so run it only against the intended project database.
