# Claim-Support Annotation Guide

See `annotation_workflow.md` for the candidate-to-benchmark lifecycle and the exact separation of system and human fields.

## Unit of annotation

Each item is one generated claim paired with one cited passage. Passage-ID validity and semantic support are separate questions: a real passage ID proves only that the passage exists.

## Labels

- `SUPPORTED`: the passage directly entails the complete claim.
- `PARTIALLY_SUPPORTED`: the passage supports only part of the claim or requires missing context.
- `UNSUPPORTED`: the passage is related but does not justify the claim.
- `CONTRADICTED`: the passage conflicts with the claim.

## Decision rules and edge cases

1. Judge the complete claim, including qualifiers, population, method, comparison, and outcome.
2. Do not infer unstated causal, comparative, or general claims from topical similarity.
3. Use `PARTIALLY_SUPPORTED` when a material part is supported but another part lacks evidence.
4. Use `CONTRADICTED` only when the passage supplies conflicting evidence, not when it is merely silent.
5. Mark malformed, empty, or missing claim/passage text invalid for annotation rather than guessing a label.
6. Negative passages from other retrieved papers must be mixed with cited passages, and model decisions must remain hidden.

## Review and adjudication

- Reviewers label independently using stable anonymized IDs.
- Preserve original ratings and notes. Store adjudication separately.
- Resolve disagreements from the displayed claim and passage only; do not expose gold labels or model decisions.
- Freeze development and test splits. Use development data only for confidence-policy selection.
