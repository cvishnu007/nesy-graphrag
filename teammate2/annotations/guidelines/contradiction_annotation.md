# Contradiction Annotation Guide

See `annotation_workflow.md` for the candidate-to-benchmark lifecycle and the exact separation of system and human fields.

## Evidence shown to reviewers

Review the two paper titles and abstracts. The model prediction, confidence, candidate score, and other reviewers' labels must remain hidden during independent annotation. Compare claims only when their conditions are sufficiently similar.

## Labels

- `CONTRADICTION`: the abstracts make incompatible claims under comparable conditions.
- `AGREEMENT`: the abstracts support compatible conclusions.
- `DIFFERENT SCOPE`: the claims concern different populations, tasks, settings, or outcomes and cannot be directly compared.
- `UNCERTAIN`: the abstract evidence is insufficient. Retain these cases for audit and exclude them from primary scoring.

## Decision rules and edge cases

1. Prefer `DIFFERENT SCOPE` when both claims may be true because their conditions differ.
2. Use `CONTRADICTION` only for incompatible conclusions, not merely different methods, terminology, or performance magnitudes.
3. Use `AGREEMENT` when the conclusions are compatible; exact wording need not match.
4. Use `UNCERTAIN` when abstracts omit the conditions or results needed for a defensible comparison.
5. Treat an unordered paper pair as one item. Paper IDs must be stored in lexical order; reversed duplicates are invalid.
6. Hard negatives and high-overlap non-contradictions must remain eligible for annotation. Do not construct the benchmark only from predicted contradictions.

## Review and adjudication

- Two reviewers independently label at least 20-30% of the pairs using stable anonymized reviewer IDs.
- Reviewers must not see predictions or confidence values before submitting their labels.
- Preserve every original label. Store the adjudicated label separately; never overwrite reviewer labels.
- Discuss disagreements using only the supplied paper evidence and record a concise adjudication reason.
- Freeze development and test splits after adjudication. Tune thresholds only on development data.
