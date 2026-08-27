# Chinmay - Phase 3 Work Log

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
