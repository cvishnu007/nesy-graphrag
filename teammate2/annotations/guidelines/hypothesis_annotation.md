# Hypothesis Annotation Guide

See `annotation_workflow.md` for the candidate-to-benchmark lifecycle and the exact separation of system and human fields.

## Review procedure

Use hypotheses produced for frozen queries, including accepted and rejected generations. Remove model confidence and feasibility labels, randomize display order, and have two reviewers score a meaningful subset. Reviewer IDs must be stable and anonymized.

## Rating rubric

Only scores `1`, `3`, and `5` are valid.

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Evidence | No support | Partial support | Clear multi-paper support |
| Novelty | Already obvious | Some new connection | Clearly non-trivial connection |
| Feasibility | Not testable | Possible with major work | Specific and practically testable |
| Specificity | Vague | Partly defined | Clear variables and expected relationship |
| Usefulness | No clear value | Potential value | Strong research value |

## Acceptance rule

Predeclare acceptance before final test review. The default PDF rule accepts a rating when evidence, feasibility, and specificity are each at least `3`, with no dimension scored `1`.

## Consistency and adjudication

- Apply the anchors literally and score every dimension independently.
- HNS is a structural graph proxy and must not be used as the human novelty rating.
- Preserve each reviewer's original scores and notes.
- Record adjudicated scores separately without overwriting original ratings.
- Report reviewer agreement only when at least two reviewers rated common hypotheses. Otherwise report insufficient data.
