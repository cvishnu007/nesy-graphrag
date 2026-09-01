# Phase 3 annotation-readiness report

Date: 2026-08-31

## Status

Phase 3 is **PARTIALLY COMPLETE** at the annotation-ready stage. Real human
annotation has not started, so agreement, adjudication, benchmark finalization,
and freezing have not been claimed.

## Pre-implementation audit

The existing schemas and tests were reviewed before implementation.

- Contradiction records use `annotations` containing `reviewer_id`, `label`, and
  required `reason`; supported labels are `CONTRADICTION`, `AGREEMENT`,
  `DIFFERENT SCOPE`, and `UNCERTAIN`.
- Claim-support records use `annotations` containing `reviewer_id`, `label`, and
  optional `notes`; supported labels are `SUPPORTED`, `PARTIALLY_SUPPORTED`,
  `UNSUPPORTED`, and `CONTRADICTED`.
- Hypothesis records use `ratings` with `reviewer_id`, optional notes, and 1/3/5
  scores for evidence, novelty, feasibility, specificity, and usefulness.
- All tasks already have a separate `adjudication` field. Existing categorical
  finalization detects disagreement and refuses to choose a label silently.
- `finalize_annotation_pool()` is the applicable draft benchmark finalizer and is
  reused. `finalize_judgments.py` applies only to retrieval judgments.
- `validate_annotation_pool()` and the validators in
  `reasoning_benchmark_io.py` remain authoritative.
- The three reasoning benchmark files were empty valid drafts before Phase 3.

No existing reviewer assignment, reviewer-isolated packet, deterministic
double-sampling, or agreement-report tool existed.

## Implemented annotation-ready workflow

`src/evaluation/phase3_annotation.py` now provides:

- strict anonymized reviewer slot validation;
- deterministic prediction-independent assignment and double sampling;
- separate blinded packets for each reviewer;
- complete-response and assignment-coverage validation;
- categorical agreement rate and Cohen's kappa where defined;
- hypothesis exact agreement plus per-dimension Cohen's kappa where defined;
- disagreement queues that preserve both original responses;
- separate adjudicator responses without overwriting reviewers;
- draft finalization through the existing `finalize_annotation_pool()` function.

The real annotation-ready artifacts are under `evaluation/phase3/`. Preparation
validated all displayed paper references against the frozen 8,850-paper corpus
and all displayed query references against the frozen query benchmark. The tool
does not read protected Phase 2 sidecars.

## Counts

| Task | Dev | Test | Total | Double annotation |
|---|---:|---:|---:|---:|
| Contradiction | 23 | 39 | 62 | 16 (25.81%) |
| Claim support | 56 | 95 | 151 | 38 (25.17%) |
| Hypothesis | 30 | 70 | 100 | 25 (25.00%) |

Current human annotation, rating, disagreement, and adjudication counts are all
zero. Agreement statistics are unavailable until actual people submit responses.

## Validation performed

- Focused Phase 3/schema tests: **39 passed in 0.75 seconds**
- Static compilation: `python -m compileall -q src tests` passed
- Six source pools: schema/blinding checks passed
- Frozen paper/query reference checks: passed during assignment preparation
- Assignment manifest: 313 unique items and 392 exact reviewer assignments
- Reviewer packet validation at the readiness checkpoint: all 12 packets passed;
  0 responses were populated at that time
- Reviewer isolation: packet assignments exactly match each reviewer's manifest
  assignment and contain no other reviewer response
- Contradiction cross-split overlap: 0
- Draft benchmark containers: valid, empty, and status `draft`
- Workspace diff whitespace check: passed; existing unrelated CRLF warnings remain

## Integrity state

- Phase 1 frozen artifacts were not modified.
- Phase 2 raw outputs, candidates, sidecars, and source pools were not modified by
  Phase 3 preparation.
- Reviewer packets contain no system predictions, confidence, feasibility,
  acceptance, HNS, graph/retrieval scores, raw generation, or hidden labels.
- Reviewer packets are separate, initially empty, and preserve stable IDs/splits.
- Draft benchmark containers remain unpopulated and unfrozen.
- No model evaluation, provider selection, or threshold tuning was performed.

## Human action still required

Assign real people privately to the anonymized reviewer slots, give each only
their own packet directory, and follow `evaluation/phase3/README.md`. After all
responses are complete, run the documented analyze command, have the independent
adjudicator resolve every disagreement, and only then run draft finalization.

## AI pilot requested after readiness preparation

A six-item AI-only qualitative pilot was added under
`results/reasoning/phase3_ai_pilot/`. It uses the first two development records
from each blinded task pool and is explicitly marked `ai_pilot_not_gold`,
`is_human_annotation: false`, and `eligible_for_benchmark_gold: false`. Protected
sidecars were not used. These pilot judgments were not inserted into human
packets, source pools, benchmarks, agreement calculations, or adjudication.

The pilot does not change the Phase 3 status: real independent human review is
still required before benchmark finalization and freezing.

### Subsequent packet-state observation

After the AI pilot was stored, a read-only integrity check found two populated
responses in `reviewer_01/contradiction_dev.json` for `C0C22508E24E7` and
`C0D27B651D266`. They were not created by the AI pilot operation and were left
unchanged. Their provenance must be confirmed before they are counted as human
annotations. Official source pools still contain zero human entries and all three
benchmark files still contain zero gold records.
