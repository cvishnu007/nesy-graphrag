# Teammate 2 Remaining Responsibilities and Implementation Plan

Date reviewed: 2026-08-30
Branch: `master`
Baseline commit before Teammate 2 changes: `6560e0c`
Authoritative specification: `C:\Users\gurleen kaur\Downloads\nesy_graphrag_teammate2_complete_guide.pdf`
Current automated verification: **247 tests passing**, Python compilation passing, and `pip check` passing.

## Executive summary

### Completion status — September 1, 2026

- Phases 1 and 2: complete.
- Phase 3: complete using AI-generated reference annotations as explicitly
  requested; the originally proposed human workflow was not performed.
- Phase 4: mandatory semantic-support integration complete.
- Phase 5: mandatory contradiction, claim-support, and hypothesis AI-reference
  evaluation complete. Results are baselines and include weak/negative results.
- Phases 6, 7, and 8: controlled NER, local-embedding, and blinded LLM
  comparisons complete without overwriting production graph/vector stores.
- Phase 9: final verification, documentation, and all seven authorized live
  integration checks complete.

All annotations and derived benchmarks are AI-generated references. They are not
human ground truth, human reviewed, or evidence of human agreement/kappa.

The reusable **evaluation infrastructure and required controlled experiments are complete**. The repository has schemas, validators, blinded AI-reference annotation, frozen benchmarks, standard metrics, semantic support, hypothesis statistics, controlled NER/embedding/LLM comparisons, serialized results, metadata, and automated tests.

The human-annotation requirement was replaced by the user's approved AI-reference methodology. This is a methodological substitution, not a claim that human work occurred. Actual results and remaining live-check state are recorded in `TEAMMATE2_FINAL_COMPLETION_REPORT.md`.

Historical remaining-work sections below are retained as the original audit trail;
their status is superseded by this September 1 update and the final report.

## What is already complete

### Environment and repository preparation

- Python 3.11.9 environment works at `venv/`.
- Pinned project dependencies and `en_core_web_sm` are installed.
- CPU PyTorch is available; CUDA is not available in the current environment.
- The branch started from the inspected local master commit.
- The suite increased from 125 baseline tests to 187 passing tests.
- No frozen corpus was regenerated and no replacement papers were downloaded.

### Offline benchmark and annotation infrastructure

- Empty/draft containers exist for contradiction, claim support, and hypothesis ratings.
- Reasoning-specific schema validation covers IDs, labels, splits, score ranges, duplicate records, external references, canonical contradiction identity, and reversed pairs.
- Annotation guides contain the PDF label definitions and five-dimension hypothesis rubric.
- The annotation workflow clearly separates system-supplied fields from human-owned fields.
- Annotation-pool tooling creates stable IDs, blinds system decisions, writes a separate system sidecar, prevents duplicates, and requires human data before finalization.
- Fixture-only benchmarks are marked as fixtures and cannot be confused with real results through runner metadata.

### Offline metrics and runner

- Classification accuracy, per-class precision/recall/F1, macro metrics, and confusion matrices are implemented.
- Candidate Recall@K, pair coverage, confidence bins, malformed/rejection coverage, and threshold sweeping are implemented.
- Claim-support classification, false acceptance, unsupported rejection, supported rate, and coverage are implemented.
- A provider-neutral semantic-support interface and deterministic multi-passage aggregation are implemented.
- Hypothesis mean/std, acceptance, observed agreement, weighted Cohen's kappa, human/model feasibility agreement, and HNS/human-novelty correlation are implemented.
- The offline runner validates benchmarks, joins separate predictions by stable ID, serializes raw results before metrics, records failures/prerequisites, and protects existing output directories.

### Existing project functionality to reuse

- `src/pipeline/contradiction.py`: graph-ranked contradiction candidate generation.
- `src/pipeline/verdicts.py`: strict contradiction verdict/confidence parsing.
- `src/pipeline/provenance.py`: stable claim/passage IDs and structural citation validation.
- `src/pipeline/review.py`: frozen-query review generation and claim/passage export source.
- `src/pipeline/hypothesis.py`: structural-hole generation and accepted/rejected audit records.
- `src/pipeline/metrics.py`: HNS structural proxy; it must remain separate from human novelty.
- Teammate 1's frozen queries, splits, IR metrics, BM25/vector/graph/hybrid implementations, and retrieval results.

## Requirement-by-requirement remaining work

| PDF requirement | Current status | What remains |
|---|---|---|
| Branch/environment preparation | PARTIAL | Verify the newly delivered frozen artifacts, configure real local Neo4j/Groq access, record exact versions/counts, and resynchronize/merge master before final handoff. |
| Freeze schemas and annotation rules | COMPLETE for infrastructure | Do not change label definitions casually. Add a benchmark README and freeze actual benchmark versions only after real annotation/adjudication. |
| Contradiction benchmark | NOT STARTED with real data | Generate a real candidate pool; label at least 50 pairs, preferably 100; include graph candidates, agreements, different-scope cases, high-overlap negatives, and hard negatives; double-label 20-30%; adjudicate; freeze dev/test. |
| Candidate-generation evaluation | READY, NOT EXECUTED | Run graph candidate generation against the gold contradiction set and report Candidate Recall@K and pair coverage. |
| Contradiction verdict evaluation | READY, NOT EXECUTED | Produce real predictions, tune confidence only on dev, freeze the threshold, run test once, and report class metrics, macro F1, accuracy, confusion matrix, confidence bins, malformed cases, rejection coverage, and `UNCERTAIN` audit counts. |
| Required contradiction comparisons | NOT EXECUTED | Compare candidate heuristic without LLM, LLM without confidence rejection, and LLM with the dev-tuned threshold. The overlap/negation rule baseline is optional. |
| Claim-support benchmark | NOT STARTED with real data | Run reviews on frozen queries, export every claim/cited passage pair, add difficult negative passages, blind decisions, collect independent human labels, adjudicate, and freeze dev/test. |
| Passage-ID versus semantic support reporting | INFRASTRUCTURE COMPLETE | On real runs, report structural provenance and semantic support separately. Never describe a valid passage ID as entailment. |
| Semantic-support provider | INTERFACE ONLY | Select at least one real NLI/scientific entailment model or strict deterministic LLM judge based on benchmark results; implement the provider adapter and record its exact model/version. |
| Semantic-support pipeline integration | MISSING | After `validate_claim_provenance`, send only structurally valid claim/passage pairs to the provider; reject or warn on unsupported/contradicted claims; retain raw decisions and rejected claims; preserve original provenance records. |
| Semantic-support comparison | NOT EXECUTED | Compare passage-ID existence only versus semantic verification on the same claim-support benchmark. If an LLM judge is used, enforce temperature 0, strict labels, bounded retries, version logging, and no gold-label access. |
| Claim-support results | NOT EXECUTED | Report per-class and macro metrics, supported-claim rate, unsupported rejection, false acceptance, confident coverage, malformed outputs, and low-confidence cases. |
| Human hypothesis benchmark | NOT STARTED with real data | Sample accepted and rejected hypotheses from frozen queries, hide model feasibility/confidence, randomize display, collect 1/3/5 ratings from two reviewers on a meaningful subset, and retain original/adjudicated ratings. |
| Hypothesis results | READY, NOT EXECUTED | Report mean/std for all five dimensions, predeclared acceptance rate, reviewer agreement, human/model feasibility agreement, and HNS/human-novelty relationship. Do not describe HNS as human novelty. |
| NER comparison | MISSING | Choose one scientific NER/concept alternative; build a separate graph version; keep embedding/LLM/queries fixed; compare concept quality, graph size, candidate recall, contradiction F1, and hypothesis ratings. |
| Embedding comparison | MISSING | Choose one stronger scientific embedding model; use a separate Chroma collection; keep corpus/queries/top-k/graph/downstream model fixed; report Teammate 1 IR metrics plus reasoning metrics. |
| LLM comparison | MISSING | Choose one available alternative to the current Groq model; freeze candidates, prompt, context, temperature, and retries; measure contradiction F1, malformed rate, support quality, hypothesis ratings, latency, calls, and cost. |
| One-change-at-a-time experiment discipline | NOT EXECUTED | Never change NER, embedding, and LLM in the same experiment. Record a configuration snapshot and experiment ID for every run. |
| Full reasoning runner | OFFLINE CORE COMPLETE; LIVE EXECUTION MISSING | Add/verify adapters that generate real task predictions from the frozen pipeline. Load only needed clients, close drivers safely, and run dev/test using frozen configuration. |
| Required result artifact set | PARTIAL | Real runs must produce metadata, contradiction/support predictions and metrics, hypothesis ratings/metrics, model comparison CSV, and failures JSONL. Current fixture outputs are not results. |
| Reproducibility metadata | PARTIAL | Add real corpus/graph/index versions, prompt hashes/version, NER/embedding/LLM/support models, thresholds, random seed, runtime, API call count, latency, cost where available, and failure counts. Never store secrets or full environment dumps. |
| Required live integration checks | BLOCKED | Run one live contradiction evaluation, one live review with semantic support, one live hypothesis evaluation, and one development model comparison. |
| Final test discipline | NOT EXECUTED | Tune only on dev, freeze prompts/models/thresholds, run final test under identical configuration, retain failures/rejections, and never repair predictions after viewing gold labels. |
| Documentation and result summaries | MISSING | Add benchmark README, README commands, verified limitations/negative results, and result summaries. Update `PROJECT_STATUS.md` only after results are real and verified. |
| Git/PR handoff | MISSING | Organize commits, fetch/merge master without `reset --hard`, rerun verification and `git diff --check`, push `reasoning-evaluation`, and provide reproduction steps and limitations in the PR. |

## Detailed remaining implementation plan

### Phase 1 - Accept and verify Teammate 1 artifacts

Goal: prove that all later experiments use the exact intended frozen state.

1. Place the supplied corpus, Chroma directory, and Neo4j artifact in their documented local locations without rebuilding them.
2. Verify hashes/version identifiers supplied by Teammate 1.
3. Record:
   - Git commit
   - corpus source/version and actual paper count
   - Chroma collection name/model/count
   - Neo4j paper/author/concept/citation counts
   - frozen query benchmark version and dev/test splits
   - current NER, embedding, LLM, prompts, thresholds, and date
4. Run read-only Chroma and Neo4j connectivity/count checks.
5. Run the existing full tests before producing candidates.

Deliverable: a verified artifact manifest under `results/reasoning/<run-id>/metadata.json` or a dedicated manifest referenced by every experiment.

### Phase 2 - Generate real blinded annotation pools

Goal: turn frozen pipeline outputs into annotation-ready records without leaking predictions.

#### Contradictions

1. Run `detect_contradictions` across frozen development/test queries.
2. Add high-overlap non-contradictions, agreements, different-scope cases, and terminology-similar hard negatives.
3. Deduplicate canonical unordered pairs.
4. Generate stable IDs and a separate system sidecar using `src.evaluation.annotation_pool`.
5. Target at least 50 total pairs; prefer 100 as specified by the PDF.

#### Claim support

1. Run reviews on frozen queries.
2. Export every generated claim and each cited passage as a separate annotation item.
3. Add difficult negative passages from other retrieved papers without changing the gold label automatically.
4. Generate the blinded pool and separate support-prediction sidecar.

#### Hypotheses

1. Run hypothesis generation on frozen queries.
2. Include both accepted and rejected generations.
3. Remove model feasibility/confidence, acceptance, and HNS from annotator-facing files.
4. Preserve displayed evidence and randomize presentation order outside stable identity.

Deliverables:

- `evaluation/annotation_pools/contradiction_dev.json`
- `evaluation/annotation_pools/contradiction_test.json`
- `evaluation/annotation_pools/claim_support_dev.json`
- `evaluation/annotation_pools/claim_support_test.json`
- `evaluation/annotation_pools/hypothesis_dev.json`
- `evaluation/annotation_pools/hypothesis_test.json`
- corresponding protected system sidecars under `results/reasoning/annotation_sidecars/`

### Phase 3 - Human annotation and benchmark freezing

Goal: create real gold data rather than treating model output as truth.

1. Assign stable anonymized reviewer IDs.
2. Independently double-label at least 20-30% of contradiction pairs and a meaningful subset of hypothesis/support items.
3. Hide model predictions, confidence, scores, HNS, feasibility, and other reviewers' labels.
4. Resolve disagreements and store adjudication separately without overwriting originals.
5. Run `finalize_annotation_pool` and reasoning benchmark validation.
6. Keep status `draft` during review.
7. Freeze benchmark version and dev/test partitions only after duplicate/reference/adjudication checks pass.

Deliverables: populated and frozen versions of the three files under `evaluation/benchmarks/`, plus a benchmark README describing creation, reviewers, sample sizes, labels, splits, and limitations.

### Phase 4 - Select and integrate semantic support verification

Goal: close the gap between structural citation validity and semantic entailment.

1. Implement provider adapters behind the existing `verify_claim_support` interface:
   - existence-only baseline adapter/result path
   - at least one real local NLI/scientific entailment model or strict LLM judge
2. Benchmark provider choices on claim-support **development** labels.
3. Select the provider and confidence threshold using development data only.
4. Integrate semantic verification after `validate_claim_provenance` in the review flow.
5. Keep original provenance, raw support decisions, unsupported/contradicted claims, malformed output, and low-confidence cases for audit.
6. Add focused integration tests using provider doubles plus one controlled live smoke check.

Likely code changes:

- extend `src/evaluation/semantic_support.py` with concrete adapter(s)
- add a small integration point in `src/pipeline/review.py` or an evaluation-only review wrapper
- extend `src/evaluation/reasoning_runner.py` to invoke only the selected task/provider
- add provider/integration tests
- pin a new direct dependency only if a selected local model genuinely requires it

### Phase 5 - Run real baseline reasoning evaluation

Goal: produce defensible development results and then one frozen test run.

1. Generate raw contradiction/support/hypothesis outputs in deterministic order.
2. Save raw outputs before computing metrics.
3. Tune contradiction and semantic-support confidence thresholds on dev only.
4. Freeze thresholds, prompts, models, and all configuration.
5. Run the final test split once under the frozen setup.
6. Produce:
   - `metadata.json`
   - `contradiction_predictions.jsonl`
   - `contradiction_metrics.json`
   - `claim_support_predictions.jsonl`
   - `claim_support_metrics.json`
   - `hypothesis_ratings.csv`
   - `hypothesis_metrics.json`
   - `failures.jsonl`
7. Report `UNCERTAIN`, malformed, rejected, and low-confidence cases separately.

### Phase 6 - Controlled NER comparison

Goal: change only concept extraction/graph construction.

1. Select one scientific NER/concept alternative based on hardware and licensing.
2. Preserve the current spaCy graph as the baseline.
3. Build a separately named graph/database from the same frozen corpus.
4. Keep embedding, LLM, prompts, queries, and thresholds fixed.
5. Compare concept quality, graph size/density, contradiction candidate Recall@K, verdict F1, and hypothesis human ratings.
6. Record indexing time, runtime, failures, and configuration.

### Phase 7 - Controlled embedding comparison

Goal: change only the embedding model/index.

1. Select one stronger scientific retrieval embedding.
2. Build a separately named Chroma collection; never overwrite `s2_papers`.
3. Keep graph, corpus, query set, top-k, LLM, prompts, and reasoning thresholds fixed.
4. Run Teammate 1's IR metrics and Teammate 2 reasoning metrics.
5. Record model ID, collection/version, indexing/runtime resources, and failures.

### Phase 8 - Controlled LLM comparison

Goal: change only the generator/judge model.

1. Select one available alternative to `openai/gpt-oss-120b`.
2. Reuse exactly the same retrieved contexts/candidates and prompt versions.
3. Fix temperature, retries, maximum tokens, and confidence policy.
4. Measure contradiction F1, malformed-output rate, semantic support, hypothesis ratings, latency, API calls, and cost.
5. Preserve raw outputs and failures for both models.

### Phase 9 - Final verification, documentation, and handoff

1. Run all required live integration checks.
2. Run:

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m compileall -q src app tests
.\venv\Scripts\python.exe -m pip check
git diff --check
```

3. Update README commands, benchmark README, limitations, and negative results.
4. Update `PROJECT_STATUS.md` only with verified completed work and real results.
5. Ensure no keys, caches, generated stores, or secret-bearing exceptions are tracked.
6. Fetch and merge current master without `git reset --hard`.
7. Rerun tests after the merge.
8. Commit coherent implementation/result batches and push `reasoning-evaluation`.
9. Open the PR with exact reproduction steps, artifact versions, known limitations, and honest statistical claims.

## Priority order once artifacts arrive

| Priority | Work | Why |
|---|---|---|
| P0 | Verify frozen artifact identity/counts | Every later result depends on using the correct corpus, graph, and index. |
| P0 | Generate real blinded dev annotation pools | Human labeling is the longest critical-path task. |
| P0 | Start contradiction and claim-support annotation | Threshold selection and provider choice cannot proceed defensibly without dev gold. |
| P1 | Implement/test the selected semantic provider adapter | Required to compare existence-only versus semantic verification. |
| P1 | Complete hypothesis sample and human ratings | Needed for human quality, agreement, and HNS relationship. |
| P1 | Freeze dev-selected settings and run final baseline test | Produces the central Teammate 2 results. |
| P2 | NER comparison | Requires rebuilding a separate graph and rerunning downstream evaluation. |
| P2 | Embedding comparison | Requires a separate index and Teammate 1 plus reasoning metrics. |
| P2 | LLM comparison | Requires frozen contexts and controlled live calls, latency, and cost tracking. |
| P3 | Final documentation, merge, and PR | Must describe only verified work and results. |

## Work that can happen in parallel

- Contradiction and claim-support human annotation can run while the semantic provider adapter is being implemented.
- Hypothesis human rating can run while contradiction thresholds are being tuned on dev.
- Metadata/prompt hashing and result packaging can be strengthened while reviewers annotate.
- NER/embedding/LLM alternatives can be shortlisted, but no comparison should be claimed until benchmarks are frozen.

## Current blockers and owners

| Blocker | Needed from | Blocks |
|---|---|---|
| Exact frozen corpus | Teammate 1 | Real pool generation and all experiments |
| Matching Chroma collection | Teammate 1 | Review/retrieval and embedding baseline |
| Matching Neo4j graph/artifact | Teammate 1 | Contradiction candidates, hypotheses, graph metrics |
| Frozen query/split/version confirmation | Teammate 1 | Controlled dev/test evaluation |
| Human reviewers | Project team | Gold labels, adjudication, agreement, hypothesis ratings |
| Real Groq credentials/model access | User/project | Live contradiction/review/hypothesis and LLM comparison |
| Semantic-provider selection | Teammate 2 after dev labels | Semantic integration and support results |

## Exact first commands after artifact restoration

First verify the current offline implementation still passes:

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m compileall -q src
.\venv\Scripts\python.exe -m pip check
```

Then verify Chroma and Neo4j counts using the supplied frozen state:

```powershell
.\venv\Scripts\python.exe -c "from src.storage.chroma_store import get_collection; print('Chroma vectors:', get_collection().count())"
.\venv\Scripts\python.exe -c "from src.storage.neo4j_store import get_driver; d=get_driver(); s=d.session(); print('Papers:', s.run('MATCH (p:Paper) RETURN count(p) AS c').single()['c']); print('CITES:', s.run('MATCH ()-[r:CITES]->() RETURN count(r) AS c').single()['c']); s.close(); d.close()"
```

After producing the real contradiction candidate export, generate the blinded development pool:

```powershell
.\venv\Scripts\python.exe -m src.evaluation.annotation_pool --task contradiction --input results\reasoning\candidates\contradiction_candidates.jsonl --annotation-output evaluation\annotation_pools\contradiction_dev.json --system-output results\reasoning\annotation_sidecars\contradiction_dev_system.json --split dev
```

Do not freeze or score the benchmark until real reviewers complete the labels and adjudication.

## Definition of remaining completion

Teammate 2 is genuinely complete only when all of the following are true:

- The three real benchmarks are populated, reviewed, adjudicated, versioned, and frozen.
- Candidate recall and contradiction verdict metrics are reported on frozen data.
- Semantic support is integrated after passage-ID validation and compared with the existence-only baseline.
- Unsupported and contradicted claims remain auditable.
- Human hypothesis ratings, acceptance, reviewer agreement, feasibility agreement, and HNS relationship are reported.
- NER, embedding, and LLM comparisons each change only one component.
- Only development data was used for thresholds/provider choices.
- Final test outputs, failures, and complete metadata are saved.
- Required live integration checks and the complete automated suite pass.
- Results, limitations, negative findings, reproduction commands, and artifact versions are documented.
- No credentials, model caches, or generated stores are committed.
- The final PR is clean, reproducible, and based on the current master branch.
