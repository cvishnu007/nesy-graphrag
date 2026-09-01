# Phase 5 AI-reference evaluation report

Date: 2026-09-01

## Scope and methodology

These evaluations use **AI-generated reference annotations**, not human ground
truth. No human review, human agreement, or human Cohen's kappa is claimed.
Development data was used for threshold selection; test data was evaluated once
with the selected threshold.

## Contradiction baseline

- Development records: 23; selected confidence threshold: 0.5.
- Test records: 39; macro-F1: 0.2924; coverage: 0.5641.
- Test contradiction F1: 0.0. This weak result is reported without inflation.
- Predictions came from protected Phase 2 system outputs that were hidden from
  Phase 3 AI-reference annotation passes.

## Semantic-support baseline

- Model: locally cached `cross-encoder/nli-deberta-v3-small`.
- No benchmark text was sent to an external service.
- Development records: 56; selected confidence threshold: 0.0; macro-F1: 0.2661.
- Test records: 95; macro-F1: 0.2855; accuracy: 0.3684; coverage: 1.0.
- Test false-acceptance rate: 0.0; unsupported-claim rejection rate: 1.0.
- The four-way mapping is declared in code: entailment maps to supported,
  contradiction to contradicted, and sufficiently strong secondary entailment
  under a neutral winner maps to partially supported.

The local NLI baseline is conservative and performs poorly on partial support;
it is a reproducible baseline, not a claim of research-grade accuracy.

## Hypothesis reference summaries

- Development: 30 hypotheses; AI-reference acceptance rate: 0.9333.
- Test: 70 hypotheses; AI-reference acceptance rate: 0.6143.
- Metrics are explicitly named reference/annotation-pass metrics. They must not
  be interpreted as human ratings or human agreement.

## Artifacts

All predictions, threshold sweeps, metrics, metadata, and failure logs are under
`results/reasoning/phase5_ai_reference_baseline/`.

## Final validation

- Focused Phase 3/reasoning/semantic regression suite: 90 passed.
- Complete project test suite: 227 passed.
- `python -m compileall -q src tests`: passed.
- `python -m pip check`: no broken requirements.
- Phase 3 integrity: 12 packets, 313 unique items, 392 assignments, zero
  null responses, 33 disagreements, and 33 consensus records.
- Final benchmark counts: contradiction 62, claim support 151, hypothesis 100.
- Both protected response substring hashes still match the pre-finalization
  values recorded in `PHASE3_AI_REFERENCE_FINALIZATION_REPORT.md`.
