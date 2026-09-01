

Package source commit: `ddc7fe3`
Final live-check commit: `ddc7fe3`
Benchmark version: `1.0-ai-reference-frozen`

## Non-human methodology declaration

Every contradiction, claim-support, and hypothesis annotation and benchmark in
this package is an **AI-generated reference annotation**. These records are not
human ground truth and were not human reviewed. No human agreement, human
Cohen's kappa, human acceptance, human feasibility agreement, or human
adjudication is claimed. Duplicate judgments and consensus are repeated AI
passes and AI-generated consensus only.

## 1. Phase 1 — artifact acceptance and verification

### What was implemented

- Added read-only artifact inspection for the cleaned corpus declaration,
  Chroma collection, Neo4j counts/connectivity, query benchmark, prompt hash,
  model configuration, and consistency relationships.
- Added strict prerequisite reporting so missing or unavailable services remain
  visible rather than silently becoming successful results.
- Saved reproducible metadata and audit reports without copying stores or
  credentials into this package.

### Files

- `tooling/evaluation/artifact_manifest.py`
- `tooling/evaluation/benchmark_io.py`
- `evaluations/phase1_verified/metadata.json`
- `reports/PHASE1_PHASE2_ARTIFACT_STATUS_AND_RUNBOOK.md`
- `reports/TEAMMATE2_SETUP_BASELINE_AUDIT.md`
- `tests/test_phase1_phase2_tooling.py`

### Method and result

Verification was read-only. The final live evidence confirmed 8,850 papers in
both Chroma and Neo4j. No corpus, Chroma store, or Neo4j store is included here.

## 2. Phase 2 — blinded evaluation-pool preparation

### What was implemented

- Exported contradiction pairs, claim/passage support items, and hypotheses
  from saved reasoning outputs.
- Generated stable task IDs and deterministic development/test pools.
- Removed protected model predictions, confidences, feasibility decisions, HNS,
  and other system-only fields from annotator-visible records.
- Stored protected decisions separately in sidecars.
- Added duplicate, reference, schema, identity, and leakage checks.

### Files

- `tooling/evaluation/reasoning_candidate_export.py`
- `tooling/evaluation/annotation_pool.py`
- `tooling/evaluation/collect_reasoning_outputs.py`
- `annotations/blinded_pools/`
- `annotations/guidelines/`
- `evaluations/candidates/`
- `evaluations/annotation_sidecars/`
- `evaluations/phase2_raw/`
- `tests/test_annotation_pool.py`
- `tests/test_phase1_phase2_tooling.py`

### Method and result

Six blinded pools were prepared: development and test files for contradiction,
claim support, and hypotheses. Protected outputs stayed outside visible records
and were not used as annotation labels.

## 3. Phase 3 — AI-reference annotation and benchmark finalization

### What was implemented

- Implemented checkpointed AI annotation that reads only blinded packet fields.
- Resumed existing checkpoints rather than restarting completed responses.
- Validated all reviewer packets and exact assignment coverage.
- Quarantined 42 invalid support responses containing a contradiction-only
  label, preserved the original packet, and regenerated only invalid fields.
- Detected duplicate-pass disagreements and generated separate AI consensus.
- Built finalized annotated pools, frozen benchmarks, provenance, and hashes.

### Files

- `tooling/evaluation/phase3_annotation.py`
- `tooling/evaluation/phase3_ai_annotation.py`
- `tooling/evaluation/reasoning_benchmark_io.py`
- `tooling/evaluation/freeze_ai_reference_benchmarks.py`
- `annotations/phase3/reviewer_packets/`
- `annotations/phase3/ai_consensus.json`
- `annotations/phase3/ai_annotated_pools/`
- `annotations/phase3/quarantine/`
- `benchmarks/contradiction_pairs.json`
- `benchmarks/claim_support.json`
- `benchmarks/hypothesis_ratings.json`
- `benchmarks/ai_reference_frozen_manifest.json`
- `reports/PHASE3_AI_REFERENCE_FINALIZATION_REPORT.md`
- `tests/test_phase3_annotation.py`
- `tests/test_phase3_ai_annotation.py`
- `tests/test_reasoning_benchmarks.py`
- `tests/test_freeze_ai_reference_benchmarks.py`

### Method and result

- 12 packets, 313 unique items, and 392/392 populated assignments.
- Zero null responses.
- 33 duplicate AI-pass disagreements and 33 AI consensus records.
- Frozen benchmark counts: contradiction 62 (23 dev/39 test), support 151
  (56/95), and hypotheses 100 (30/70).
- Two pre-existing responses were preserved byte-for-byte:
  - `C0C22508E24E7` — SHA-256
    `70f28eccc2d3a622c6a4fd322a9dee1f0a148fbe00e01c509488523864835694`.
  - `C0D27B651D266` — SHA-256
    `4bab4295da094478d0fad5877366c8d919d2d7960f86cf2c949c15079075ee9d`.

## 4. Phase 4 — semantic claim-support verification

### What was implemented

- Added provider-neutral semantic support after structural passage-ID
  validation.
- Added `SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, and `CONTRADICTED`
  decisions, confidence validation, and deterministic passage aggregation.
- Integrated an offline cached NLI provider,
  `cross-encoder/nli-deberta-v3-small`.
- Kept per-passage scores, reasons, errors, unsupported claims, contradicted
  claims, and malformed/low-confidence decisions auditable.
- Preserved original review behavior when semantic checking is disabled.

### Files

- `tooling/evaluation/semantic_support.py`
- `tooling/pipeline/review.py`
- `tooling/utils/config.py`
- `evaluations/final_ai_reference/support_comparison/`
- `tests/test_semantic_support.py`
- `tests/test_provenance.py`

### Method and result

Semantic support runs only after passage identities are structurally validated.
It improved the frozen-test existence-only macro-F1 from `0.0940` to `0.2855`
and accuracy from `0.2316` to `0.3684`. False acceptance decreased from `1.0`
to `0.0`, and unsupported rejection increased from `0.0` to `1.0`. The
partial-support F1 remained `0.0`, which is retained as a negative finding.

## 5. Phase 5 — contradiction, claim-support, and hypothesis evaluation

### What was implemented

- Implemented standard accuracy, per-class precision/recall/F1, macro metrics,
  confusion matrices, confidence bins, coverage, malformed/rejected counts, and
  development-only threshold selection.
- Added contradiction candidate Recall@K and full-pool coverage.
- Added claim-support metrics including false acceptance and unsupported
  rejection.
- Added five-dimension hypothesis summaries for evidence, novelty, feasibility,
  specificity, and usefulness, plus acceptance and AI-pass agreement.
- Saved raw predictions, ratings, failures, metadata, thresholds, and commands.

### Files

- `tooling/evaluation/classification_metrics.py`
- `tooling/evaluation/contradiction_runner.py`
- `tooling/evaluation/contradiction_candidate_evaluator.py`
- `tooling/evaluation/claim_support_metrics.py`
- `tooling/evaluation/hypothesis_metrics.py`
- `tooling/evaluation/reasoning_runner.py`
- `evaluations/phase5_ai_reference_baseline/`
- `evaluations/final_ai_reference/contradiction_candidate/`
- `evaluations/final_ai_reference/hypothesis/`
- relevant metric and end-to-end tests under `tests/`

### Important results

- Candidate test full-pool recall `1.0`, Recall@10 `0.2308`, and Recall@20
  `0.4872`. Full-pool recall is expected because reference pairs originated in
  candidate pools; cutoff recall is the meaningful ranking result.
- Baseline contradiction test macro-F1 `0.2924`, coverage `0.5641`, and
  contradiction F1 `0.0`.
- Semantic support test macro-F1 `0.2855`, accuracy `0.3684`, and coverage `1.0`.
- Hypothesis test: 70 hypotheses, aggregate mean `3.2419/5`, and
  hypothesis-level acceptance `0.6143`.
- Feasibility-model agreement and HNS relationship remain
  `insufficient_data`; no value was fabricated.

## 6. Controlled NER comparison

### What was implemented and method

Compared the current `en_core_web_sm` entity-plus-noun-chunk extractor with a
fixed scientific-term/model-token pattern extractor on the same deterministic
500-document sample. Only the extractor changed. The production graph was not
rebuilt or overwritten.

### Files and results

- `tooling/evaluation/ner_comparison.py`
- `evaluations/final_ai_reference/model_comparisons/ner/`
- `tests/test_ner_comparison.py`
- Baseline: 10,223 total concepts, 7,260 unique, mean 20.446/document.
- Alternative: 1,624 total, 160 unique, mean 3.248/document.
- Mean document Jaccard: `0.0280`.

The pattern extractor is conservative and is not claimed to be a superior
scientific NER model.

## 7. Controlled embedding comparison

### What was implemented and method

Compared locally cached SPECTER and MiniLM using identical query text, document
text, judged-ID candidate universes, cosine similarity, and cutoffs. Only the
embedding model changed. The production Chroma collection was not overwritten.
Model/provider selection was performed on development data before frozen test.

### Files and results

- `tooling/evaluation/embedding_comparison.py`
- `tooling/evaluation/ir_metrics.py`
- `benchmarks/retrieval_queries_judged.json`
- `evaluations/final_ai_reference/model_comparisons/embeddings/`
- `tests/test_embedding_comparison.py`
- Frozen test MiniLM NDCG@10 `0.7434`, Recall@10 `0.2936`, MAP `0.4250`.
- Frozen test SPECTER NDCG@10 `0.3568`, Recall@10 `0.1274`, MAP `0.1678`.

The retrieval judgments are provisional non-human references, not verified
human relevance judgments.

## 8. Controlled LLM comparison

### What was implemented and method

- Added strict blinded JSON prompts and checkpointed/resumable predictions.
- Excluded reference labels, reasons, annotators, adjudication, and protected
  sidecars from prompts.
- Compared `qwen/qwen3.6-27b` and `qwen/qwen3.8-27b` with identical records,
  prompts, temperature 0, 1,000-token budget, disabled reasoning, threshold
  0.5, and batch size 4. Only the model ID changed.
- Added bounded parse retries and failure auditing after diagnosing an initial
  hidden-reasoning token exhaustion.

### Files and results

- `tooling/evaluation/llm_comparison.py`
- `tooling/utils/groq_client.py`
- `evaluations/final_ai_reference/model_comparisons/llm/`
- `tests/test_llm_comparison.py`
- `tests/test_groq_client.py`
- Frozen test 27B accuracy/macro-F1 `0.7436/0.4943`.
- Frozen test 8B accuracy/macro-F1 `0.7179/0.4666`.
- Coverage was `1.0` for both. Both missed the single test contradiction, so
  contradiction F1 was `0.0` for both.

## 9. Live integration checks

### What was implemented

Added structured, non-destructive checks for Chroma, Neo4j, cached local NLI,
Groq, semantic review, contradiction, and hypothesis. Failures are stored as
data and drivers are closed in `finally` blocks.

### Files and final results

- `tooling/evaluation/live_completion_checks.py`
- `evaluations/final_ai_reference/live_checks/`
- `tests/test_live_completion_checks.py`
- Final result: 7 passed, 0 failed, 0 not run,
  `all_required_passed: true`.
- Chroma and Neo4j each reported 8,850 papers.
- Review: 2/2 citations verified and 5/5 claims structurally grounded, with
  semantic checking enabled.
- Contradiction: one valid `DIFFERENT SCOPE` verdict at confidence `0.88`.
- Hypothesis: one valid `MEDIUM`-feasibility result.

## 10. Testing and validation

- Added tests for schemas, blinding, leakage, packet completion, AI checkpoints,
  consensus, freezing, metrics, semantic support, comparisons, Groq controls,
  live failures, and review integration.
- Final focused completion suite: 30 passed.
- Final complete project suite: 247 passed.
- `python -m compileall -q src tests`: passed.
- `python -m pip check`: no broken requirements.
- All 12 packets, 392 assignments, 33 consensus records, frozen hashes, and two
  protected byte-level response hashes were revalidated.

Relevant test snapshots are under `tests/`; synthetic fixtures are explicitly
marked fixture-only and are not experimental results.

## 11. Reproducibility and metadata

- Every final experiment saves model/provider IDs, inputs and hashes where
  applicable, split, thresholds, controlled variables, runtimes, failures, and
  exact reproduction commands.
- `evaluations/final_ai_reference/completion_manifest.json` records the final
  requirement matrix and Phase 3 counts.
- `MANIFEST.json` records every file in this package with its size and SHA-256.
- `reproduce.ps1` contains the final verification and live-check commands.
- No secrets, full environment dumps, databases, corpus exports, or caches are
  included.

## 12. Git and handoff work

- Work was based on master ancestry at `6560e0c` and completed through
  `ddc7fe3`.
- Work was organized into scoped commits for design, benchmark freezing,
  candidate evaluation, semantic comparison, hypothesis reporting, NER,
  embeddings, LLMs, live evidence, and final documentation.
- Staged-path and secret scans excluded `.env`, credentials, model caches,
  corpus data, Chroma, Neo4j stores, and virtual environments.
- The pre-existing corrupted `.env.example` and stray accidental file were not
  staged, committed, modified, or copied into this package.
- This packaging task will show its exact staged contents before any commit.
- Nothing is pushed by this workflow.

## Final limitations and non-claims

- AI reference annotations may share model biases and do not establish human
  validity.
- The contradiction class is extremely small in frozen test data.
- Semantic NLI remains weak for partially supported claims.
- The NER alternative is a conservative rule pattern, not validated scientific
  NER.
- Retrieval references are provisional and non-human.
- No statistical-significance, model-training, or large-corpus scaling result
  is claimed.
