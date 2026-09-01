# Deliverables

This folder is a focused, self-contained evidence package for Gurleen's
Teammate 2 responsibilities in NeSy-GraphRAG. It contains only the implemented
evaluation tooling, AI-reference annotation artifacts, frozen benchmarks,
experiment outputs, relevant tests, metadata, failure logs, reproduction
commands, and reports needed to understand and audit the work.

## Methodology boundary

All contradiction, claim-support, and hypothesis annotations and benchmarks in
this package are **AI-generated reference annotations**. They are not human
ground truth and were not human reviewed. No human agreement, human Cohen's
kappa, human acceptance, or human adjudication is claimed. Any agreement or
consensus record refers only to repeated AI annotation passes.

## Contents

- `gurleen.md` — complete responsibility-by-responsibility record.
- `tooling/` — snapshots of the Teammate 2 implementation modules.
- `annotations/` — guidelines, blinded pools, reviewer packets, AI consensus,
  quarantine evidence, and finalized AI-annotated pools.
- `benchmarks/` — the three frozen reasoning benchmarks, frozen manifest,
  benchmark README, and the provisional retrieval-reference input needed for
  the embedding comparison.
- `evaluations/` — Phase 1/2 verification outputs, protected sidecars,
  candidates, baseline reasoning results, final comparisons, failures,
  metadata, and live-check evidence.
- `tests/` — relevant Teammate 2 tests and small synthetic fixtures.
- `reports/` — final reports and historical readiness/runbook documentation.
- `reproduce.ps1` — verification commands to run from the repository root.
- `MANIFEST.json` — exact packaged file list, sizes, SHA-256 hashes, source
  commit, and exclusions.

## Scope and exclusions

The package intentionally excludes:

- `.git`, `.env`, `.env.example`, credentials, and API keys;
- `venv`, `.tools`, Python caches, pytest caches, and model caches;
- `data/s2_clean.json`, other corpus exports, and full corpus data;
- Chroma and Neo4j database/store directories;
- unrelated Teammate 1 or user files;
- the pre-existing corrupted `.env.example` and stray accidental file.

The blinded and annotated evaluation records necessarily retain the paper
titles, abstracts, claims, and passages used by the benchmarks. These are
bounded benchmark evidence records, not a copy of the source corpus.

## Reproduction model

This is an evidence package, not a duplicate application checkout. Run
`reproduce.ps1` from the parent repository with its existing Python environment
and configured services. The commands use the production source tree while the
snapshots under `tooling/` make the exact Teammate 2 implementation easy to
review in isolation.

The package was assembled from source commit `ddc7fe3`. No push is performed by
the packaging workflow.
