# Phase 3 AI-reference annotation record

Status: **complete AI-generated reference annotation; not human reviewed**

All 392 reviewer-assignment slots are populated. The completed artifacts are:

- `ai_consensus.json`: 33 AI-generated consensus records for duplicate-pass disagreements;
- `ai_annotated_pools/`: six complete dev/test pools;
- `../benchmarks/contradiction_pairs.json`: 62-record draft AI-reference benchmark;
- `../benchmarks/claim_support.json`: 151-record draft AI-reference benchmark;
- `../benchmarks/hypothesis_ratings.json`: 100-record draft AI-reference benchmark.

These are AI-generated reference annotations, not human ground truth. No human
review, human agreement, human Cohen's kappa, or human adjudication was performed
or claimed. Per-response and artifact provenance records the AI models used.

The historical human workflow below was prepared before the AI-reference
methodology was selected. It was not executed and must not be used to describe
the completed artifacts.

## Historical human-workflow instructions (not performed)

These files were generated only from the six blinded Phase 2 pools. The workflow
does not read `results/reasoning/annotation_sidecars/` and must never be given
access to that directory during independent review.

## Reviewer slots

- `reviewer_01`: independent reviewer slot
- `reviewer_02`: independent reviewer slot
- `reviewer_03`: adjudicator slot

These are anonymized slots, not invented people. Assign actual humans to the slots
outside the repository and keep the mapping private. Do not add names, emails, or
other identifying information to annotation or benchmark files. Each person must
use the same slot across every task.

## Deterministic assignments

Seed: `phase3-double-annotation-v1`

Requested double-annotation fraction: 25% for each task.

| Task | Total | Double annotated | Actual fraction | Dev double | Test double |
|---|---:|---:|---:|---:|---:|
| Contradiction | 62 | 16 | 25.81% | 4 | 12 |
| Claim support | 151 | 38 | 25.17% | 15 | 23 |
| Hypothesis | 100 | 25 | 25.00% | 9 | 16 |

Sampling is a fixed-seed hash of task and stable item ID. It does not use model
predictions, confidence, graph scores, retrieval scores, or sidecar information.
Stable IDs and dev/test membership are unchanged.

## Independent annotation procedure

1. Give each reviewer only their own directory under `reviewer_packets/`.
2. Do not give either reviewer the other reviewer's files or any protected
   sidecar.
3. For every record, replace `response: null` with a response matching the
   packet's `response_schema`.
4. Use an ISO-8601 timestamp such as `2026-08-31T14:30:00+05:30`.
5. Set the packet-level status to `complete` when every response is filled.
6. Validate each completed packet:

```powershell
.\venv\Scripts\python.exe -m src.evaluation.phase3_annotation validate-packet --packet evaluation\phase3\reviewer_packets\reviewer_01\contradiction_dev.json --require-complete
```

Repeat for every packet assigned to each reviewer. The supported labels and rating
anchors are defined in `evaluation/guidelines/`; difficult-negative passages do
not receive an automatic label.

## Agreement and adjudication

After every assigned response is complete, run:

```powershell
.\venv\Scripts\python.exe -m src.evaluation.phase3_annotation analyze --manifest evaluation\phase3\assignment_manifest.json --packet-dir evaluation\phase3\reviewer_packets --output-dir evaluation\phase3\human_review
```

This validates exact assignment coverage, calculates agreement from independent
human responses only, and creates `human_review/adjudications.json`. Only
`reviewer_03` should receive that adjudication file. It contains both original
responses for each disagreement; fill its `response` without changing either
original response.

## Draft benchmark finalization

Only after all required adjudication responses are complete, run:

```powershell
.\venv\Scripts\python.exe -m src.evaluation.phase3_annotation finalize --pool-dir evaluation\annotation_pools --manifest evaluation\phase3\assignment_manifest.json --packet-dir evaluation\phase3\reviewer_packets --adjudications evaluation\phase3\human_review\adjudications.json --output-dir evaluation\phase3\annotated_pools --benchmark-dir evaluation\benchmarks
```

This reuses `finalize_annotation_pool()` and produces **draft** benchmarks. It
refuses missing reviewer responses, assignment mismatches, malformed reviewer
IDs, invalid labels/scores, and unresolved disagreements. Do not change benchmark
status to `frozen` until the complete Phase 3 checklist and reference audit pass.
