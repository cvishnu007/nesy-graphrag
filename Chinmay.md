# Chinmay — Local generation and evaluation work log

Updated: September 14, 2026. **Paused at the user’s request; partial checkpoint pushed.** Branch: `feat/local-evaluation`, based on the latest `origin/phase3` (`880d76e`); local `master` was already current at `ee27d00`.

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
- Added development-only partial-support calibration over saved NLI probabilities. Test partial-support F1 increased **0 → 0.125**, macro F1 **0.2855 → 0.3190**, but false acceptance increased **0 → 0.03125**. This experimental threshold was not adopted in production. Prior test weakness motivated the experiment, so confirmatory improvement claims need a fresh untouched test set.
- Added a separate, licensed SciFact claim/evidence benchmark and reproducible evaluation CLI. It preserves complete rationale sets and separates official train/local development (955 items) from official dev/local test (338 items, 122 contradicted). The fixed NLI model achieved contradiction F1 **0.6286**, two-gold-class macro F1 **0.5143**, and overall accuracy **0.3580**. It evaluates biomedical entailment, not graph paper-pair detection. The frozen in-domain paper-pair benchmark still has one positive test contradiction.
- Added and extended automated tests for provider selection/retries/fallbacks, pipeline routing, BM25, frozen-pool integrity, matched contexts, packaged paths/hashes, development-only calibration, SciFact conversion and label-blinded evaluation. The current full suite passes **306 tests**; compile, dependency consistency and diff-whitespace checks also pass. One upstream spaCy/Click deprecation warning remains.
- Added `docs/LOCAL_EVALUATION.md`, refreshed `PROJECT_STATUS.md`, and updated README/setup instructions with local startup, safe rebuilds, portable reproduction and interpretation limits. Tightened live checks so empty or invalid structured generations cannot silently count as successful pipeline evidence.

- Fixed the live expanded-graph rebuild failure: the previous single-transaction `DETACH DELETE` exceeded Neo4j's transaction-memory limit. Rebuilds now delete in 1,000-node transactions and consume the result before reporting success (Neo4j 5.23+ syntax). The dedicated-database reset guard remains required.
- Corrected expanded retrieval protocol capture to read fusion weights from the evaluation configuration module and record graph limits, hop depth and production weights; added a regression check. Matched RAG records the same retrieval settings.

- Added `store_audit.py` and a stale-evidence regression test; the audit checks all titles/abstracts as well as paper-ID parity and NER completion.

- The first grading attempt stopped on malformed JSON after eight development-only judgments. Preserved it under `abandoned_dev_format_v1`; added schema-constrained Ollama output, bounded batch length/grades, and a regression test. A second development-only trial confirmed schema validity but spent most generation time repeating IDs/prose. Preserved and discarded its 24 dev labels too. Final protocol 1.2 returns ordered integer grades mapped to saved batch IDs; no rationale is requested or invented. All protocol changes preceded test-query grading, and no retrieval settings changed.
- Streamlit startup and its three-mode selector passed the live AppTest with no exceptions; the loopback demo previously ran at `http://127.0.0.1:8501`. Application-mode generation checks remain pending.

- Added a rank-only overlap diagnostic: production fusion returns the same ordered top 20 as vector retrieval for 15 of 20 queries. No labels were used and no retrieval settings were changed after this check.

- Added a results-CSV whitespace attribute so Git accepts standard CSV CRLF endings without rewriting frozen artifact bytes; checks still detect trailing spaces and blank lines.

## Pause, push and laptop cleanup — September 14

The user requested that implementation stop, the half-done changes be pushed, and downloaded laptop runtime resources such as Ollama be removed. No further evaluation work is being started.

Completed code is retained in local commits `579de55` and `185659d`, with this pause checkpoint added afterward on `feat/local-evaluation`. The push targets that feature branch; `master` and `phase3` are not being merged or overwritten.

**Exact unfinished state:** the final grading protocol saved **778 / 1,145** AI-reference judgments (all 347 development candidates, plus 431 test candidates through part of Q014). **367 remain.** The run lost its Ollama connection and its watcher exited before producing a complete benchmark. Expanded retrieval scoring, all four significance comparisons, the full matched-RAG experiment, and live backend generation checks did **not run**. Streamlit startup/mode-selection and a direct Ollama completion were verified earlier; those do not substitute for the unfinished generation checks. Frozen candidates, rankings, protocols, raw responses, partial checkpoints and abandoned development-only format trials are preserved under `results/retrieval/expanded_20260913/`, with an explicit `PAUSED.json` marker. No partial retrieval score is reported as a final result.

Cleanup performed:

- Stopped the Ollama launch service and confirmed the background grading/watcher/Streamlit/Neo4j jobs were no longer running.
- Uninstalled session-installed Homebrew formulae: **ollama, neo4j, cypher-shell, openjdk@21, mlx-c, mlx, little-cms2**. Checked installed dependents before removing the extra libraries.
- Removed the session-created `.ollama` directory (Qwen weights/cache/keys), downloaded DeBERTa NLI cache and locks, SciFact source download/extraction, dedicated Neo4j data/config/logs, and matching Homebrew download caches.
- Restored the original `.env` and original **10,000 raw / 8,850 clean / 8,850 NER** JSON files byte-for-byte from the pre-session backup. Unrelated environment changes were checked before restoration; none were overwritten.
- Restored a compact Chroma collection with exactly the original **8,850 IDs, documents and metadata**, reusing the original vectors without model inference. Chroma cosine reinsertion causes small float32 renormalization differences; all vectors were checked within absolute tolerance `1e-6` (maximum observed `9.54e-7`). Removed the expanded index, expanded JSON corpus, expansion checkpoint and now-redundant temporary backup.
- Preserved the pre-existing Python virtual environment and SPECTER model cache. Existing shared Homebrew libraries that were upgraded during installation were retained to avoid breaking other software; their names are recorded in `results/local/runtime_cleanup_20260914.json`.
- Preserved small diagnostic logs and the completed evaluation evidence in Git. The 47,627-paper store audit describes the verified September 13 snapshot, **not the laptop’s state after cleanup**. The demo is now stopped and its services/models are uninstalled.
- Updated README, project status, the local runbook, and the execution checklist to show the pause and cleanup. After cleanup, **306 tests passed** (one upstream spaCy/Click warning); compilation and `pip check` passed. Checks confirmed all seven installed formulae and the model directories absent, ports 11434/7687/8501 closed, and exact 8,850-ID parity across the restored clean/NER/Chroma data. Delivery verified: checkpoint commit **`4b434e0`** was successfully pushed to **`origin/feat/local-evaluation`** on September 14. The committed pool blobs match all four frozen input hashes, so the partial grading checkpoint remains resumable. This documentation follow-up is included on the same branch.

## Resume later

1. The saved retrieval pool can finish grading and score cached rankings on another configured Ollama machine; restore the exact settings in `judge_protocol.json`, then run the documented `grade` and `evaluate` stages in the existing directory. Keep the original relevance criteria and retrieval settings fixed; 431 test judgments already exist.
2. Run the predeclared significance comparisons after all judgments are complete. Do not treat the AI references as human ground truth.
3. Rebuild/reacquire matching expanded stores and reinstall services before live demo or matched-RAG work. The currently restored laptop data has only 8,850 papers and cannot be described as the removed 47,627-paper snapshot.
4. Complete local backend generation checks before the held-out matched-RAG comparison, then update this work log with actual outcomes.
5. Independent retrieval/answer review, stronger in-domain paper-pair contradiction coverage and expert hypothesis evaluation remain research work. The separate SciFact results and partial-support calibration already completed are retained with their limitations.

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
