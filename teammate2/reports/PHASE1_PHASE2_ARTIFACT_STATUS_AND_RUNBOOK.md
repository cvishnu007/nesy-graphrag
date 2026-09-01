# Teammate 2 Phase 1/2 artifact status and runbook

Date: 2026-08-30

This document records only evidence observed in this repository. It does not claim
that missing Neo4j state was verified, and it does not treat test fixtures as real
benchmark data.

## Current outcome

Phase 1 is **partially complete**. The supplied corpus, local Chroma store, query
benchmark, code revision, models, prompts, and thresholds were inspected without
rebuilding them. Neo4j remains a hard prerequisite and is explicitly marked
unavailable in the manifest.

Phase 2's collection/export/blinding path is implemented and covered by offline
tests. The six real annotation pools have **not** been generated because doing so
without the frozen Neo4j artifact (and, for support/hypotheses, Groq access) would
either fail or produce non-authoritative data.

## Verified frozen-state evidence

The machine-readable manifest is
`results/reasoning/phase1_20260830/metadata.json`.

| Item | Observed value | Status |
|---|---|---|
| Git commit | `6560e0cc06b666d1d15a24a3832e7232a325251e` | Recorded |
| Corpus | `data/s2_clean.json` | Verified read-only |
| Corpus SHA-256 | `16bb235f4a16d6d0f3c07f69461274ea89f2a9cdf69194664220d997bb5deecf` | Locally computed |
| Corpus papers | 8,850 unique non-empty IDs; years 2020-2025 | Verified |
| Chroma directory | `data/chromadb` | Verified read-only |
| Chroma directory fingerprint | `923383168d346ca1530980aeb9082393be1cba1c16ff3e6649d08b6a1c4d98666` | Locally computed from 20 files |
| Configured collection | `s2_papers`, 8,850 records, cosine space | Verified |
| Other local collection | `arxiv_papers`, 9,990 records | Observed; not selected for this run |
| Embedding model | `allenai-specter` | Recorded from configuration; Chroma metadata does not independently attest the model |
| Query benchmark | version `0.2-draft`, status `judgments_pending_human_review` | Verified |
| Query benchmark SHA-256 | `204bd998786f1b368e3a191fe6ff955f7f325879f42497d46e274cacbf599874` | Locally computed |
| Query splits | 6 development, 14 test | Verified |
| NER | `en_core_web_sm` 3.8.0 | Verified locally |
| Primary/fallback LLM | `openai/gpt-oss-120b`; `llama-3.1-8b-instant` | Recorded from configuration |
| Neo4j | credentials incomplete; local port 7687 not listening | **Blocked/unverified** |
| Declared graph counts | 42,937 concepts; 7,203 citations | Benchmark declaration only; **not accepted as live verification** |

The corpus paper count matches both `s2_papers` and the benchmark declaration.
The manifest also records the current prompt hash and all configured contradiction,
hypothesis, and semantic-support thresholds. No secrets are written to it.

## What to obtain from Teammate 1

Do not run the repository's graph creation/reset path as a substitute. Request:

1. The exact Neo4j dump/backup or documented immutable graph artifact.
2. Its SHA-256 (or the team's agreed version identifier), Neo4j server version,
   database name, and restoration instructions.
3. Teammate 1's expected counts for `Paper`, `Author`, `Concept`, and `CITES`.
4. The exact corpus and Chroma identifiers/hashes they supplied, so the locally
   computed values above can be checked against their source-of-truth values.
5. If their restoration procedure depends on it, the matching `s2_ner.json`;
   that file is not present locally. It must not be regenerated for this evaluation.

Restore the artifact into the matching Neo4j version/database, start it, and set
`NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` locally. Keep
`NEO4J_ALLOW_RESET=false`. Credentials must not be committed or copied into a
manifest.

## Phase 2 implementation now available

- `src/evaluation/collect_reasoning_outputs.py` runs existing contradiction,
  review, and hypothesis functions over a frozen dev/test split. It stores raw
  pipeline outputs and explicit sanitized failures.
- `src/evaluation/reasoning_candidate_export.py` converts those native outputs
  into unlabeled contradiction, claim-support, or hypothesis candidates. It
  canonicalizes contradiction pairs, removes duplicates, exports every cited
  passage separately, can include other-paper difficult support candidates, and
  retains accepted plus rejected hypotheses.
- `src/evaluation/annotation_pool.py` creates stable IDs, validates records, and
  separates annotator-visible content from protected system predictions. Human
  labels/ratings are left empty and predictions are not leaked into blinded files.
- `src/evaluation/artifact_manifest.py` performs repeatable read-only Phase 1
  verification and records a manifest that can be referenced by experiments.
- `tests/test_phase1_phase2_tooling.py` covers manifest evidence, deterministic
  collection, sanitized failures, canonical/deduplicated pairs, difficult support
  candidates, accepted/rejected hypotheses, and absence of invented labels.

The contradiction pool must be reviewed for the required mixture of agreements,
different-scope cases, high-overlap non-contradictions, and terminology-similar
hard negatives. Candidate source/type may be supplied by the system, but the gold
contradiction label must only come from a human annotator. The final real target is
at least 50 unique pairs, with 100 preferred.

## Exact execution order after Neo4j arrives

First create a new, fully verified manifest. Do not overwrite the partial record:

```powershell
.\venv\Scripts\python.exe -m src.evaluation.artifact_manifest --output results\reasoning\phase1_verified\metadata.json
```

Confirm that `verification_status` is `verified`, the four Neo4j counts match
Teammate 1's expected counts, and no collection was rebuilt. Then collect each
split. Contradictions require Neo4j; support and hypotheses additionally require
the configured Groq credential used by the frozen pipeline.

```powershell
.\venv\Scripts\python.exe -m src.evaluation.collect_reasoning_outputs --tasks contradiction support hypothesis --split dev --output-dir results\reasoning\phase2_raw\dev --top-k 10
.\venv\Scripts\python.exe -m src.evaluation.collect_reasoning_outputs --tasks contradiction support hypothesis --split test --output-dir results\reasoning\phase2_raw\test --top-k 10
```

For each split, export candidates and produce blinded files plus protected
sidecars. The following is the development sequence; repeat with `dev` changed to
`test` and `--split test`.

```powershell
.\venv\Scripts\python.exe -m src.evaluation.reasoning_candidate_export --task contradiction --input results\reasoning\phase2_raw\dev\contradiction_pipeline_outputs.jsonl --output results\reasoning\candidates\contradiction_dev.jsonl
.\venv\Scripts\python.exe -m src.evaluation.annotation_pool --task contradiction --input results\reasoning\candidates\contradiction_dev.jsonl --annotation-output evaluation\annotation_pools\contradiction_dev.json --system-output results\reasoning\annotation_sidecars\contradiction_dev_system.json --split dev

.\venv\Scripts\python.exe -m src.evaluation.reasoning_candidate_export --task support --input results\reasoning\phase2_raw\dev\support_pipeline_outputs.jsonl --output results\reasoning\candidates\claim_support_dev.jsonl --negatives-per-claim 1
.\venv\Scripts\python.exe -m src.evaluation.annotation_pool --task support --input results\reasoning\candidates\claim_support_dev.jsonl --annotation-output evaluation\annotation_pools\claim_support_dev.json --system-output results\reasoning\annotation_sidecars\claim_support_dev_system.json --split dev

.\venv\Scripts\python.exe -m src.evaluation.reasoning_candidate_export --task hypothesis --input results\reasoning\phase2_raw\dev\hypothesis_pipeline_outputs.jsonl --output results\reasoning\candidates\hypothesis_dev.jsonl
.\venv\Scripts\python.exe -m src.evaluation.annotation_pool --task hypothesis --input results\reasoning\candidates\hypothesis_dev.jsonl --annotation-output evaluation\annotation_pools\hypothesis_dev.json --system-output results\reasoning\annotation_sidecars\hypothesis_dev_system.json --split dev
```

Before annotation, inspect candidate counts, diversity, provenance, and the
collection `failures.jsonl`. Do not silently accept partial task output. Do not
open protected sidecars in an annotator workflow. Randomize only presentation
order in the annotation interface; stable identity remains unchanged.

## Verification performed after these changes

- Full suite: **194 passed** (`.\venv\Scripts\python.exe -m pytest`)
- Compilation: passed (`.\venv\Scripts\python.exe -m compileall src`)
- Dependencies: `No broken requirements found` (`.\venv\Scripts\python.exe -m pip check`)

No corpus, Chroma collection, Neo4j graph, retrieval implementation, human label,
or experimental result was created or modified during this work.
