# Reasoning benchmarks

Benchmark version: `1.0-ai-reference-frozen`

Finalization date: 2026-09-01

Status: **frozen AI-generated reference benchmarks; not human reviewed**

## Benchmark files

| Task | Records | File |
|---|---:|---|
| Contradiction | 62 | `contradiction_pairs.json` |
| Claim support | 151 | `claim_support.json` |
| Hypothesis | 100 | `hypothesis_ratings.json` |

The source pools contain 23/39 contradiction, 56/95 claim-support, and
30/70 hypothesis development/test records. Contradiction pairs are canonical and
have no development/test overlap.

## Annotation methodology

These files contain **AI-generated reference annotations**, not human ground
truth. AI annotation passes used only blinded visible task fields and did not see
protected system predictions, confidence values, graph scores, retrieval scores,
or Phase 2 sidecars. Duplicate-pass disagreements were resolved by a separate
AI-generated consensus pass while preserving both original pass responses.

Per-response and artifact metadata records the AI models used. Artifact
provenance explicitly sets these fields to false:

- `human_ground_truth`;
- `independent_human_review`;
- `human_agreement_calculated`;
- `human_cohen_kappa_calculated`.

No human review, human agreement, human Cohen's kappa, or human adjudication was
performed or claimed. Reviewer IDs are annotation-pass slots, not people.

## Provenance

- Phase 1 verified manifest: `results/reasoning/phase1_verified/metadata.json`
- Frozen corpus: `data/s2_clean.json`, 8,850 papers
- Phase 2 blinded source pools: `evaluation/annotation_pools/`
- Phase 3 assignment seed: `phase3-double-annotation-v1`
- Complete packets and consensus: `evaluation/phase3/`
- Complete annotated pools: `evaluation/phase3/ai_annotated_pools/`
- Finalization report: `PHASE3_AI_REFERENCE_FINALIZATION_REPORT.md`
- Frozen manifest: `evaluation/benchmarks/ai_reference_frozen_manifest.json`

## Labels

- Contradiction: `CONTRADICTION`, `AGREEMENT`, `DIFFERENT SCOPE`, `UNCERTAIN`
- Claim support: `SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, `CONTRADICTED`
- Hypothesis dimensions: evidence, novelty, feasibility, specificity, and
  usefulness, each scored `1`, `3`, or `5`

## Limitations

The benchmarks are frozen for reproducible project evaluation because the user
explicitly approved AI references as the substitute for human annotation. This
does not make them expert human gold. Results must identify the reference source
as AI-generated and must not be described as human-rated or human-validated
performance.
