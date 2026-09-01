# Teammate 2 Phase 2 completion report

Date: 2026-08-31

## Outcome

All six required annotation pools now exist. The previously missing test-side
claim-support and hypothesis generations, plus test contradiction LLM enrichment,
were produced with the explicitly approved revised experiment configuration.
Every individual pool passes schema, blinding, empty-human-field, stable-ID, and
sidecar correspondence validation. The full repository test suite and requested
environment checks pass.

Phase 2 is **COMPLETE**. The three canonical contradiction pair IDs that occurred
in both splits were retained unchanged in development and removed only from the
test pool and its matching protected test sidecar. Cross-split contradiction
overlap is now zero. No replacement pairs were generated because the combined
contradiction set remains above the required minimum of 50. Phase 3 was not
started during this correction.

The frozen corpus, Chroma directory, Neo4j graph, query benchmark, original model
configuration, and Phase 1 manifest were not rebuilt, overwritten, or modified.
The authoritative manifest remains:
`results/reasoning/phase1_verified/metadata.json`.

## Model configuration and provenance

The frozen primary model is `openai/gpt-oss-120b`; its full pipeline request
returned HTTP 429. The frozen fallback is `llama-3.1-8b-instant`; an earlier
attempt returned `model_not_found`. Neither frozen setting was edited.

The missing test work used this explicit, per-process revised configuration:

- primary: `openai/gpt-oss-20b`
- fallback: `openai/gpt-oss-20b`
- configuration label: `phase2-test-revised-gpt-oss-20b`
- revised configuration: `true`

This provenance is recorded in the raw records, collection summaries, exported
candidates, and protected sidecars. The two revised collection runs processed all
14 test queries with zero collection failures. Development sidecars predate this
provenance extension and therefore represent the legacy frozen Phase 1
configuration; they are not retroactively attributed to a response model.

## Annotation pools

| Pool | Records | Generation/configuration | Validation |
|---|---:|---|---|
| `contradiction_dev.json` | 23 | Legacy frozen configuration | Passed individually |
| `contradiction_test.json` | 39 | Revised 20B LLM sidecar; three dev overlaps removed | Passed individually |
| `claim_support_dev.json` | 56 | Legacy frozen configuration | Passed individually |
| `claim_support_test.json` | 95 | Revised 20B configuration | Passed individually |
| `hypothesis_dev.json` | 30 | Legacy frozen configuration | Passed individually |
| `hypothesis_test.json` | 70 | Revised 20B configuration | Passed individually |

Test claim-support composition:

- 33 provenance-grounded claims and 2 claims rejected by existing validation
- 60 cited claim/passage items
- 35 deterministic difficult-negative passage items
- 95 annotation records total

Test hypothesis composition:

- 47 accepted pipeline generations
- 23 rejected pipeline generations
- 70 annotation records total

For malformed rejected hypothesis output, the exporter uses the already-existing
graph structural-hole candidate text rather than inventing hypothesis text. The
malformed raw generation and rejection details remain protected in the sidecar.

The annotator-facing test contradiction pool retains 39 canonical IDs. The three
overlapping IDs `C670B0F1D9D37`, `C8F3CEAC2F3FF`, and `CC1ECCBE101E1` remain in
development and were removed from test together with their matching protected
test sidecar records. The remaining protected test sidecar retains revised 20B
predictions. The previous graph-only sidecar is retained as raw provenance at
`results/reasoning/phase2_raw/test/contradiction_graph_only_sidecar.jsonl`.

## Validation results

All six pools passed individual validation:

- required schema and task-specific fields are valid;
- IDs are unique within each pool;
- contradiction pairs are canonical unordered pairs;
- every pool record maps to exactly one record in its matching protected sidecar;
- record splits agree with their pool split;
- annotator files expose no prediction/verdict, confidence, feasibility,
  acceptance, HNS, graph/retrieval score, raw generation, or hidden system label;
- human annotations and ratings are empty, and adjudication is null.

Aggregate split validation originally found these three dev/test contradiction
overlaps:

| Pair ID | Paper 1 | Paper 2 |
|---|---|---|
| `C670B0F1D9D37` | Generalization and Representational Limits of Graph Neural Networks | Capturing Molecular Interactions in Graph Neural Networks: A Case Study in Multi-Component Phase Equilibrium |
| `C8F3CEAC2F3FF` | How Framelets Enhance Graph Neural Networks | On the Bottleneck of Graph Neural Networks and Its Practical Implications |
| `CC1ECCBE101E1` | On the Bottleneck of Graph Neural Networks and Its Practical Implications | Graph Neural Network: Current State of Art, Challenges and Applications |

All three were retained in development and removed from test. The final counts are
23 development pairs and 39 test pairs: 62 unique contradiction pairs in total.
Global dev/test overlap is zero, all remaining stable IDs are unchanged, and each
annotation record has exactly one matching protected sidecar record.

## Source and test changes

Modified `src/evaluation/collect_reasoning_outputs.py`:

- added explicit model/fallback overrides that require a configuration label;
- applies overrides only to the collector process without editing frozen config;
- records generation configuration in outputs and summaries;
- rejects caught LLM failure text instead of exporting it as annotation data.

Modified `src/evaluation/reasoning_candidate_export.py`:

- propagates generation configuration into protected candidate metadata;
- preserves rejected hypotheses while using existing graph candidate text when
  malformed output provides no parsed hypothesis.

Modified `src/evaluation/annotation_pool.py`:

- treats generation configuration as system-only data;
- places that provenance only in protected sidecars.

Modified tests:

- `tests/test_phase1_phase2_tooling.py`
- `tests/test_annotation_pool.py`

Coverage now includes caught LLM failures, revised-configuration provenance,
malformed rejected-hypothesis fallback, and blinding of model configuration.

## Generated and preserved outputs

- revised raw test support/hypothesis outputs under
  `results/reasoning/phase2_raw/test_model/`
- revised raw contradiction LLM outputs under
  `results/reasoning/phase2_raw/test_contradiction_llm/`
- candidate exports under `results/reasoning/candidates/`
- six pools under `evaluation/annotation_pools/`
- six corresponding sidecars under
  `results/reasoning/annotation_sidecars/`
- original test graph-only contradiction sidecar under
  `results/reasoning/phase2_raw/test/`

The redundant `results/reasoning/phase2_staging/` validation copy was deleted
after exact identity checks; it was temporary and is not needed for recovery.

## Verification

- Focused Teammate 2 suite: **46 passed in 49.31 seconds**
- Full suite: **201 passed in 12.25 seconds**
- `python -m compileall -q src app tests`: passed
- `python -m pip check`: passed; no broken requirements

## Completion status

All six annotation pools pass individual validation, cross-split contradiction
overlap is zero, human-owned fields remain empty, blinding passes, and all pool
records have exact protected-sidecar correspondence. The final pool counts are:

- contradiction: 23 dev / 39 test
- claim support: 56 dev / 95 test
- hypothesis: 30 dev / 70 test

Phase 2 can now be marked **COMPLETE**. No Phase 3 annotation, benchmark freezing,
model evaluation, or threshold tuning was performed as part of Phase 2.
