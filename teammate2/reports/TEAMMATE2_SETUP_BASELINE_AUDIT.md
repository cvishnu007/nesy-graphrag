# Teammate 2 Setup and Baseline Audit

Date: 2026-08-30 (Asia/Calcutta)
Branch: `reasoning-evaluation`
Commit: `6560e0c` (`origin/master`, `master`, and the working branch pointed to the same commit at inspection time)
Authoritative specification: `C:\Users\gurleen kaur\Downloads\nesy_graphrag_teammate2_complete_guide.pdf`
Specification SHA-256: `23C6B24E61E5F5D426641E6E0E1147BA123049903B5D5FFDF4B4960CD50A0441`

## Scope and change boundary

This pass performed environment setup, read-only repository inspection, PDF extraction/render review, and baseline verification only. No Teammate 2 feature, benchmark, metric, integration, test, or source-code requirement was implemented. No existing tracked source file was modified.

Environment-only changes:

- Installed a self-contained Python 3.11.9 runtime under `.tools/` because the only system interpreter was Python 3.14.0 and the existing `venv` referenced a removed Python 3.11.9 executable.
- Preserved the broken original environment at `.tools/broken-venv-original`.
- Created the repository-standard ignored `venv/` using Python 3.11.9.
- Installed all pinned packages from `requirements.txt`, plus `en_core_web_sm` 3.8.0.
- Installed `pypdf`, `pdfplumber`, and `pymupdf` only for specification inspection.
- Created ignored `.env` from `.env.example`; values remain placeholders and no secrets were added.
- Kept installer/cache and PDF-review artifacts under `.tools/`.

## Baseline verification result

| Check | Result | Evidence/notes |
|---|---|---|
| Git synchronization state | PASS (local state) | Working branch `reasoning-evaluation`; HEAD `6560e0c`; local `master` and `origin/master` resolved to the same inspected commit. No `git pull` was performed because network synchronization was not necessary to establish the local baseline. |
| Required Python | PASS | `venv\Scripts\python.exe`: Python 3.11.9. |
| Pinned dependencies | PASS | `pip check`: `No broken requirements found`. |
| Core imports | PASS | pytest, torch, pandas, spaCy, ChromaDB, Neo4j, Groq, and Streamlit imported successfully. |
| spaCy model | PASS | `en_core_web_sm` 3.8.0 installed and discoverable. |
| Compile check | PASS | `python -m compileall -q src app tests`, exit 0. |
| Unit suite | PASS | 125 tests passed in 23.29 seconds. |
| Compute | CPU only | PyTorch `2.12.1+cpu`; CUDA unavailable. This is valid for setup but model experiments will be slower. |
| `.env` | PARTIAL | Created from template, but Neo4j/Groq/Semantic Scholar secrets are still placeholders. |
| Frozen corpus | BLOCKED | `data/` is absent, so the documented 8,850-paper corpus was not locally verified. |
| Chroma | BLOCKED | `data/chromadb/` is absent; collection and count cannot be verified. |
| Neo4j | BLOCKED | `127.0.0.1:7687` is not accepting connections; graph counts cannot be verified. |
| Groq/live smoke tests | BLOCKED | No configured Groq key and no local stores; contradiction and hypothesis live calls were not fabricated or attempted. |
| PDF completeness | PASS | All 13 pages were text-extracted and rendered; every page was visually inspected. |

Important repository drift: `SETUP.md` says the repository collects 39 tests, but the current branch collects and passes 125. `README.md`/`PROJECT_STATUS.md` also contain older “not implemented” statements that conflict with the retrieval-evaluation files now present. Treat actual code/tests/results as the baseline and update documentation only during an approved implementation/handoff phase.

## Requirement-by-requirement gap analysis

Status meanings: **Exists** = directly reusable current implementation; **Partial** = related capability exists but does not satisfy the PDF; **Missing** = no conforming implementation/artifact found; **Blocked** = requires credentials, stores, human annotation, or Teammate 1 input.

### 1. Responsibility, finish condition, and scope

**PDF requirements**

- Measure whether contradiction, claim-support, and hypothesis outputs are correct, supported, useful, and reproducible.
- Freeze all three benchmarks; implement standard metrics and semantic evidence checking; finish controlled NER, embedding, and LLM comparisons; pass tests; save reproducible outputs and metadata.
- Provide unit/integration tests, limitations, saved outputs, and a clean PR.
- Do not take ownership of BM25/retrieval benchmark implementation, production UI, CI, scaling infrastructure, final report assembly, or PDF/full-text ingestion.
- Work in this order: environment; schemas/guides; benchmarks; metrics; semantic verifier; baseline evaluation; one-change-at-a-time model comparisons; tests/docs/handoff.

**Existing/reuse**: retrieval benchmarking and BM25 already exist under `src/evaluation/`, `src/pipeline/bm25_retrieval.py`, benchmark files under `evaluation/benchmarks/`, and frozen retrieval outputs under `results/retrieval/`. These are Teammate 1 assets, not work to recreate. Core reasoning modules exist under `src/pipeline/`.

**Missing/new**: all reasoning-evaluation artifacts described below. No Teammate 2 completion condition is currently met as a complete package.

**Dependency on Teammate 1**: frozen query set/splits, retrieval judgments/metrics, and fixed retrieval outputs/configuration.

**One-day priority**: follow only stage 1 from the PDF after approval—schemas, annotation guides, validators, and a tiny auditable seed set. The PDF itself estimates 14–23 focused days overall, so full completion in one day is not credible.

### 2. Safe Git/environment and baseline records

**PDF requirements**

- Base `reasoning-evaluation` on updated master; preserve unfinished changes safely; follow `SETUP.md`.
- Prepare Python 3.11, dependencies, spaCy model, Chroma, Neo4j, Groq, frozen corpus, and the project's fixed models/configuration.
- Run unit tests plus one live contradiction and one live hypothesis smoke test.
- Record Git commit/date, corpus/Chroma/Neo4j counts, prompts, LLM, embedding model, and thresholds.

**Existing/reuse**: `SETUP.md`, `.env.example`, `requirements.txt`, `src/utils/config.py`, `src/pipeline/prompts.py`, `src/utils/groq_client.py`.

**Current result**: Python/dependencies/spaCy/tests/compilation pass; commit/date/model defaults/thresholds are inspectable. Corpus, Chroma, Neo4j, Groq, and live smoke checks are blocked as listed above.

**Missing/new**: a reproducible run metadata artifact for reasoning evaluation; real local secret configuration supplied by the user; local corpus/index/graph state.

**Priority**: P0 unblockers before any live implementation: credentials, Neo4j service, and the exact frozen corpus/index/graph versions.

### 3. Required files, schemas, and configuration

**PDF requirements**

- Create three benchmarks: `contradiction_pairs.json`, `claim_support.json`, `hypothesis_ratings.json` under `evaluation/benchmarks/` (never `data/`).
- Create three annotation guides under `evaluation/guidelines/`.
- Create `classification_metrics.py`, `contradiction_runner.py`, `claim_support_metrics.py`, `semantic_support.py`, `hypothesis_metrics.py`, and `reasoning_runner.py` under `src/evaluation/`.
- Create four named test modules and `results/reasoning/.gitkeep`.
- Add six validated settings/defaults: three benchmark paths, results directory, semantic-support model, and minimum confidence; add examples to `.env.example`; pin any new direct dependency.

**Existing/reuse**: `src/evaluation/__init__.py`, retrieval-specific `benchmark_io.py`, `ir_metrics.py`, `retrieval_runner.py`, `candidate_pool.py`, and `significance.py` provide patterns for validation, deterministic runners, output writing, and statistics.

**Missing/new**: every PDF-named file and all six settings are absent. Retrieval `benchmark_io.py` is schema-specific and should be reused as a design pattern or generalized carefully, not treated as a drop-in reasoning validator.

**Dependency on Teammate 1**: directory conventions, query IDs/splits, corpus identity, runner conventions, and retrieval metrics.

**Priority**: P0 after approval: schemas/guides/config validation; P1: pure metrics; P2: runners/integration.

### 4. Contradiction benchmark

**PDF requirements**

- Prefer at least 100 labeled pairs; 50 is the small minimum.
- Include graph-generator candidates, high-overlap non-contradictions, agreements, different-scope pairs, and terminology-similar hard negatives; do not label only predicted contradictions.
- Labels/rules: `CONTRADICTION`, `AGREEMENT`, `DIFFERENT SCOPE`, `UNCERTAIN`; keep uncertain cases for audit but exclude them from primary scoring.
- Store version, stable pair ID, split, paper IDs, label, reason, annotator IDs, and adjudication state.
- Canonically sort paper IDs for stable unordered pair identity; hide predictions/confidence during annotation; double-label 20–30% independently; adjudicate without losing audit data; freeze dev/test; tune only on dev.

**Existing/reuse**: `src/pipeline/contradiction.py::detect_contradictions` creates graph-ranked candidates; `src/pipeline/verdicts.py::parse_contradiction_response` parses the three model verdicts/confidence; `src/pipeline/prompts.py::build_contradiction_prompt`; frozen query IDs/splits from Teammate 1.

**Partial**: current contradiction parsing/candidate logic exists and has unit coverage, but it is operational pipeline logic, not a labeled benchmark or blinded annotation workflow.

**Missing/new**: benchmark JSON, annotation guide/tooling, sampling and canonical deduplication process, human labels, reviewer IDs, adjudication records, frozen splits, and reasoning-specific schema validation.

**Dependency on Teammate 1**: frozen paper corpus, graph, frozen queries/splits, and retrievable paper metadata/abstracts.

**Priority**: P0 schema/guide and seed candidates; human labeling volume is the critical-path task and cannot honestly be completed by code alone in one day.

### 5. Contradiction metrics and experiments

**PDF requirements**

- Evaluate candidate generation separately using Candidate Recall@K and pair coverage.
- Evaluate verdict classification using class precision/recall/F1, positive-label precision/recall/F1, macro F1, accuracy, confusion matrix, coverage, and confidence bins.
- Implement pure functions for confusion matrix, precision/recall/F1, macro F1, accuracy, candidate recall@K, and confidence bins.
- Sweep confidence 0.50–0.90 on dev, freeze the selected threshold, then evaluate test; report rejection/coverage trade-off.
- Compare candidate heuristic without LLM, LLM without rejection, and LLM with tuned rejection; optional overlap/negation rule baseline.
- Treat unknown/malformed output as failure; report uncertain gold separately; deduplicate unordered pairs; explicitly define zero-positive behavior.

**Existing/reuse**: `parse_contradiction_response` already represents malformed output as invalid/unknown rather than silently mapping it; `src/evaluation/ir_metrics.py` demonstrates pure metric style and edge-case handling; `src/evaluation/significance.py` offers paired analysis patterns; current configuration already has `CONTRADICTION_MIN_CONFIDENCE=0.70` but it was not selected by the required new benchmark.

**Missing/new**: classification metric module, candidate-stage scoring, confidence bins/coverage, threshold sweep/freeze metadata, three required comparison runs, reasoning result files, and all specified edge-case tests.

**Dependency on Teammate 1**: retrieval/query splits and possibly paired significance conventions; BM25 work is explicitly outside Teammate 2 scope.

**Priority**: P1 once a valid dev benchmark exists. Pure metrics/tests are achievable within a day; defensible results are not without labels/live stores.

### 6. Claim-support benchmark

**PDF requirements**

- Run reviews on frozen queries; export every generated claim and cited passage; add difficult negatives from other retrieved papers; hide model decisions; label independently.
- Labels: `SUPPORTED`, `PARTIALLY SUPPORTED`, `UNSUPPORTED`, `CONTRADICTED` using the PDF definitions.
- Store item ID, split, query ID, claim, passage ID/text, paper ID, label, and notes.
- Report support precision/recall/F1, macro F1, supported-claim rate, unsupported-claim rejection rate, false acceptance rate, and confident-decision coverage.
- Report passage-ID validity separately from semantic meaning.

**Existing/reuse**: `src/pipeline/provenance.py::validate_claim_provenance` validates passage IDs and retains accepted/rejected claims; review prompt/orchestrator can generate claim-passage pairs; `src/pipeline/metrics.py::compute_provenance_ts` provides structural-provenance diagnostics.

**Partial**: structural provenance and its tests are strong, but the repository explicitly does not verify entailment.

**Missing/new**: claim-support JSON and guide, frozen exports/hard negatives, human labels/adjudication, semantic support metrics, and required result summaries.

**Dependency on Teammate 1**: frozen queries/splits and fixed retrieved contexts so support comparisons are controlled.

**Priority**: P0 schema/guide/export design; P1 small blinded seed set; full human benchmark exceeds one day.

### 7. Semantic support verification and safe integration

**PDF requirements**

- Provide a common `verify_claim_support(claim, passages)` result with label, confidence, per-passage decisions, validity, and model ID.
- Run passage-ID validation first; send only validated pairs; separate claim/evidence input; reject or warn on unsupported/contradicted claims; retain rejected claims/raw decisions; never replace provenance records.
- Select an affordable local NLI/scientific entailment model or strict structured LLM judge based on benchmark evidence.
- Compare passage-ID existence baseline against semantic verification.
- If using an LLM judge: temperature 0, strict labels, bounded retries, model/version logs, no gold-label access.
- Test direct entailment, insufficient related text, negation, deterministic multi-passage behavior, missing evidence, malformed output, and documented low-confidence policy.

**Existing/reuse**: provenance validation/records, strict prompt patterns, `groq_chat_with_retry`, model configuration, parser/audit patterns from contradiction and hypothesis code.

**Missing/new**: semantic verifier, model selection experiment, aggregation policy, support parser, threshold setting, safe pipeline integration, audit output, comparison, and all named tests.

**Dependency on Teammate 1**: fixed retrieved context and query splits; baseline passage-ID/provenance implementation already exists in shared pipeline code.

**Priority**: P1 architecture and deterministic interface/tests; model download/benchmark selection and integration are P2 and depend on labels/hardware. CPU-only execution is a schedule risk.

### 8. Human hypothesis evaluation

**PDF requirements**

- Sample accepted and rejected hypotheses from frozen queries; remove model feasibility/confidence; randomize display; double-review a meaningful subset.
- Score 1/3/5 on evidence, novelty, feasibility, specificity, and usefulness using the exact rubric anchors.
- Report mean and standard deviation per dimension, predeclared acceptance rate, human/model feasibility agreement, reviewer agreement, and HNS/human-novelty relationship.
- Anonymize reviewers with stable IDs and preserve original ratings alongside adjudication.
- Predeclare acceptance (example: evidence/feasibility/specificity each >=3 and no dimension =1).

**Existing/reuse**: `src/pipeline/hypothesis.py` generates, parses, scores, accepts/rejects, and preserves audit information; `src/pipeline/metrics.py::compute_hns` computes the structural proxy; current tests cover hypothesis parsing and candidate scoring.

**Partial**: generation, structural scoring, and acceptance logic exist, but no human benchmark or rubric metrics exist. HNS must not be reported as human novelty.

**Missing/new**: rating JSON/guide, blinded randomized review sample, reviewer/adjudication data, rubric validation/aggregation, agreement statistic, HNS correlation analysis, and required tests/results.

**Dependency on Teammate 1**: frozen queries and fixed retrieved evidence; human reviewers are an external dependency.

**Priority**: P0 predeclare rubric/schema; P1 create small sample; full double review cannot be fabricated and likely exceeds one day.

### 9. Controlled model comparisons

**PDF requirements**

- Benchmark first; never change NER, embeddings, and LLM in the same run.
- NER: spaCy versus one scientific alternative; separate Neo4j graph; measure concept quality, graph size, candidate recall, contradiction F1, and hypothesis ratings.
- Embeddings: SPECTER versus one stronger scientific model; separate Chroma collections; fix corpus/query/top-k/graph/downstream model; use Teammate 1 retrieval metrics plus reasoning metrics.
- LLM: current Groq model versus one alternative; fix prompt/context/temperature/retries/candidates; measure contradiction F1, malformed rate, claim support, hypothesis ratings, latency, and cost.
- Complete one reliable comparison per family rather than multiple incomplete runs.

**Existing/reuse**: spaCy NER, SPECTER embedding defaults, Groq primary/fallback, Chroma collection configuration, graph/index builders, Teammate 1 retrieval runner/metrics/results.

**Missing/new**: selected alternatives and justification, isolated graph/index naming/versioning, experiment matrix/config snapshots, concept-quality measurement, reasoning results, latency/API/cost capture, and model comparison CSV.

**Dependency on Teammate 1**: retrieval metrics, frozen queries/splits/corpus, tuned retrieval baseline, consistent retrieved contexts.

**Priority**: P3. Do not spend the one day on model swaps before benchmarks/metrics exist. The PDF budgets 3–6 days for this stage alone.

### 10. Unified reasoning evaluation runner and outputs

**PDF requirements**

- CLI: `python -m src.evaluation.reasoning_runner --tasks contradiction support hypothesis --split ... --output-dir ...`.
- Validate benchmark schemas/versions; lazy-load clients; deterministic order; save raw output before metrics; preserve malformed/rejected/low-confidence cases; compute item/aggregate metrics; write metadata/failures; close clients safely.
- Produce metadata, contradiction prediction/metrics, claim-support prediction/metrics, hypothesis ratings/metrics, model comparison, and failures files with the exact documented names.
- Metadata must include commit/benchmark/corpus/graph/index/prompt/model/threshold/config/seed/runtime/API/failure data and exclude keys, full environment dumps, and secret-bearing exception text.

**Existing/reuse**: `src/evaluation/retrieval_runner.py` is the closest runner template; `results/retrieval/evaluation_tuned_test/` demonstrates existing artifact conventions; config, prompt builders, and clients are reusable.

**Missing/new**: entire reasoning runner, task runners, schemas, output directory/files, redaction-safe failure capture, metadata/version hashes, deterministic raw-before-metrics workflow, and resource cleanup coverage.

**Dependency on Teammate 1**: runner conventions, frozen query split, corpus/index identifiers, retrieval results and metadata.

**Priority**: P2 after schemas and pure metrics. A skeleton without real benchmark inputs would not satisfy the PDF.

### 11. Tests, integrations, and final experiment discipline

**PDF requirements**

- Unit tests: known confusion/F1; missing classes/empty inputs; duplicate pairs; unknown verdict; support parsing/thresholds; multi-passage aggregation; hypothesis rubric validation/aggregation; duplicate IDs/invalid labels/missing fields/bad splits.
- Integration checks: one live contradiction evaluation, one live review with support decisions, one live hypothesis evaluation, and one development model comparison.
- Run pytest, compileall, pip check, and a development reasoning-runner smoke test.
- Tune only on dev; freeze prompts/models before test; keep final configurations identical; report all failures/rejections; never manually repair after seeing labels.

**Existing/reuse**: 125 current tests pass, including current contradiction parser, provenance, hypothesis, retrieval benchmarks, metrics, and runners; strict pytest configuration and fixtures are reusable.

**Missing/new**: all reasoning-specific named unit and integration tests plus development/final runs. Current 125 passing tests do not cover the new PDF requirements.

**Blocked**: live checks require credentials, corpus, Chroma, and Neo4j.

**Priority**: P0 write tests alongside approved schema/metric work; P2 integration after services/data are available.

### 12. Results, documentation, Git handoff, and definition of done

**PDF requirements**

- Summarize contradiction candidate/class metrics/coverage; support F1/false acceptance/rates/coverage; human hypothesis scores/acceptance/agreement/HNS relationship; model quality/malformed/latency/cost.
- Update annotation guides, benchmark README, README commands, verified-only PROJECT_STATUS, direct dependencies, `.env.example`, limitations, and negative results.
- Use coherent commits, merge updated master without `reset --hard`, rerun tests/diff checks, push branch, and provide reproduction steps/limitations in the PR.
- Definition of done includes frozen labels/guides, implemented metrics/verifier, audited unsupported claims, completed human ratings/agreement, isolated model comparisons, dev-only tuning, saved test metadata, all tests passing, and no committed credentials/stores.

**Existing/reuse**: README/SETUP/PROJECT_STATUS structure, Teammate 1 work log/results, clean current tracked tree before this Markdown report.

**Missing/new**: every reasoning result summary and verified documentation/handoff update. These must wait until work and experiments genuinely exist.

**Priority**: P3/final. Do not pre-claim results or edit PROJECT_STATUS early.

## One-day execution recommendation (after explicit approval)

The authoritative PDF estimates 14–23 focused working days. With one day, prioritize a defensible foundation rather than pretending the complete specification is done:

1. **P0 — Unblock inputs (owner/user + Teammate 1):** provide the frozen corpus/Chroma/Neo4j state, valid local credentials, and confirm Teammate 1 query/split/result versions.
2. **P0 — Freeze schemas and guides:** create the three benchmark schemas, label definitions, blinded/adjudication rules, duplicate-ID rules, and predeclared hypothesis acceptance policy.
3. **P0 — Add schema validators/tests:** reject duplicate IDs/pairs, invalid labels, missing fields, invalid splits, and malformed versions.
4. **P1 — Add pure metrics/tests:** classification metrics, candidate recall/coverage, claim-support metrics, hypothesis aggregates/agreement inputs, and explicit empty/zero-positive rules.
5. **P1 — Seed, do not fabricate:** create a small clearly marked development seed only from real frozen data and human judgments; do not call it the 50/100-pair finished benchmark.
6. **Defer:** semantic model selection/integration, full annotation, live final evaluation, all three model-family comparisons, final results, docs claims, and PR handoff until prerequisites and time exist.

## Approval gate

No Teammate 2 implementation has started. Implementation should begin only after the user approves a defined first-day slice and supplies/authorizes the missing live-service and frozen-data prerequisites.
