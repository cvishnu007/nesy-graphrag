# Agent Instructions

Read `AGENT_HANDOFF.md` before modifying this repository. It records the Phase 1 scope, verified data/store state, incomplete evaluation, safe commands, and non-goals.

- Preserve existing ingestion, NER, Chroma, Neo4j, and evaluation artifacts. These stages are resumable; do not delete data to restart them.
- Treat `NEO4J_ALLOW_RESET=true` as destructive. Enable it only for an explicitly requested dedicated-graph rebuild.
- Keep secrets in the ignored `.env`; never print or commit them.
- PDF/full-text ingestion and model training are out of the current scope unless the user explicitly reopens them. On 2026-11-14 the user reopened training for one component only: the SciERC entity extractor (`src/ingestion/train_scierc_ner.py`). No other model training is in scope.
- `Chinmay.md`, `devesh.md`, and `VISHNU.md` are historical contributor logs. Use `AGENT_HANDOFF.md` and `PROJECT_STATUS.md` for current state.
