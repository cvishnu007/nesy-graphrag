# Reasoning Annotation Workflow

This workflow applies to contradiction, claim-support, and hypothesis-quality annotation. Benchmark schemas, annotation pools, system sidecars, and evaluation results are different artifacts and must never be merged casually.

## Lifecycle

1. **Candidate examples** — generated from the exact frozen pipeline artifacts. Candidates may contain system predictions and internal ranking scores.
2. **Blinded annotation pool** — created by `src.evaluation.annotation_pool`. Stable IDs and source evidence are retained; predictions, confidence, acceptance decisions, HNS, and candidate scores are removed.
3. **Independent human annotation** — reviewers fill only the human-owned fields defined below. No system sidecar or other reviewer response is visible.
4. **Adjudication** — disagreements are resolved using the source evidence. Original annotations remain unchanged and adjudication is stored separately.
5. **Validated benchmark** — `finalize_annotation_pool` requires human data, converts the pool to the benchmark schema, and invokes the reasoning benchmark validator.
6. **Evaluation** — model predictions remain separate from gold benchmark files and are joined only by stable item ID in the reasoning runner.

## System-supplied versus human-supplied fields

### Contradiction

System supplies to the blinded pool:

- `pair_id`, `split`
- canonical `paper1_id`, `paper2_id`
- both titles and abstracts
- empty `annotations` and `adjudication` fields

Human reviewers supply:

- `reviewer_id`, `label`, and `reason` in `annotations`
- adjudicated `label`, `reason`, and adjudicator identity when reviewers disagree

System-only sidecar fields include prediction/verdict, confidence, and candidate score. They must not be shown during annotation.

### Claim support

System supplies to the blinded pool:

- `item_id`, `split`, `query_id`
- claim, passage ID/text, and paper ID
- empty `annotations` and `adjudication` fields

Human reviewers supply:

- `reviewer_id`, semantic-support `label`, and optional notes
- adjudicated label/notes when reviewers disagree

System-only sidecar fields include semantic prediction, confidence, and provider/model ID. Passage-ID existence is not a human semantic-support label.

### Hypothesis quality

System supplies to the blinded pool:

- `hypothesis_id`, `split`, `query_id`, hypothesis text, and displayed evidence
- empty `ratings` and `adjudication` fields

Human reviewers supply:

- `reviewer_id`
- 1/3/5 scores for evidence, novelty, feasibility, specificity, and usefulness
- optional notes and separate adjudication

System-only sidecar fields include model feasibility, acceptance status, and HNS. HNS must never prefill or replace human novelty.

## Stable IDs, splits, and duplicates

- IDs are deterministic SHA-256-derived identifiers based only on immutable source identity, not prediction or label.
- Contradiction IDs use canonical lexically ordered paper IDs, so reversed pairs receive the same identity and are rejected as duplicates.
- Claim-support IDs use query ID, normalized claim text, and passage ID.
- Hypothesis IDs use query ID and normalized hypothesis text.
- Every record must use exactly one of `train`, `dev`, or `test`. Pool generation never changes a supplied split silently.
- Duplicate identities, duplicate IDs, malformed records, unblinded fields, and missing human data during finalization are hard failures.

## Test fixtures

Files below `tests/fixtures/reasoning/` are marked `fixture_only: true`. Their labels and predictions exist solely to test software behavior. They are not benchmark data and must never be cited as evaluation results.

## Phase 3 reviewer-isolated files

`src.evaluation.phase3_annotation` prepares deterministic assignments and separate
packets under `evaluation/phase3/reviewer_packets/<reviewer_id>/`. Each reviewer
receives only their own directory. Responses are stored in that reviewer's packet;
another reviewer's responses and protected system sidecars are never copied into
it. Use only stable anonymized IDs matching `reviewer_01`, `reviewer_02`, and so
on. Keep any mapping to real people outside the repository.

Completed responses require an ISO-8601 timestamp. Agreement analysis reads only
the completed human packets and creates a separate adjudication file for the
adjudicator. It never uses model/system predictions as a reviewer or tie-breaker.
