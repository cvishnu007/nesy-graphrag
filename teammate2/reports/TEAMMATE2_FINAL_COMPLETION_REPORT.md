# Teammate 2 final completion report

Date: 2026-09-01

## Methodology boundary

The contradiction, claim-support, and hypothesis annotations and benchmarks are
**AI-generated reference annotations**. They are not human ground truth and
were not human reviewed. No human acceptance, human reviewer agreement, human
Cohen's kappa, human feasibility agreement, or human adjudication is claimed.
The user explicitly approved substituting this methodology for the requested
human-annotation deliverables.

## Completion matrix

| Requirement | Status | Evidence |
|---|---|---|
| Three populated, adjudicated, versioned, frozen benchmarks | Complete with approved AI-reference substitution | `evaluation/benchmarks/ai_reference_frozen_manifest.json`; counts 62/151/100 |
| Candidate recall and contradiction verdict metrics | Complete | `results/reasoning/final_ai_reference/contradiction_candidate/`; baseline and controlled LLM confusion matrices |
| Semantic support after passage-ID validation and existence comparison | Complete | `src/pipeline/review.py`; `results/reasoning/final_ai_reference/support_comparison/` |
| Unsupported/contradicted claims auditable | Complete | Semantic prediction streams and `support_audit.jsonl`; provenance output retains rejected decisions |
| Hypothesis evidence, novelty, feasibility, specificity, usefulness, acceptance and relationships | Complete with approved AI-reference substitution | `results/reasoning/final_ai_reference/hypothesis/`; unavailable paired relationships explicitly say `insufficient_data` |
| NER, embedding, and LLM one-change comparisons | Complete | `results/reasoning/final_ai_reference/model_comparisons/` |
| Dev-only thresholds/provider choices | Complete | Frozen dev protocols and metadata; test uses the same settings |
| Raw outputs, failures, metadata, versions and commands | Complete | Per-experiment JSON/JSONL/CSV, `failures.jsonl`, and `metadata.json` |
| Automated tests and static/dependency validation | Complete | 30 focused tests and 247 full-suite tests passed; compile and `pip check` passed |
| Live integrations | Complete | All seven read-only checks pass: Chroma, Neo4j, local NLI, Groq, semantic review, contradiction, and hypothesis |
| Documentation, limitations, negative findings and reproduction | Complete | This report, `gurleen.md`, project status, result metadata, and reproduction script |
| Credential/cache/store hygiene | Complete for scoped work | No credentials, model caches, corpus, Chroma, or Neo4j stores are included in scoped commits |

## Frozen benchmark and Phase 3 integrity

- Benchmark version: `1.0-ai-reference-frozen`.
- Contradiction: 62 records (23 development, 39 test).
- Claim support: 151 records (56 development, 95 test).
- Hypothesis: 100 records (30 development, 70 test).
- Reviewer packets: 12.
- Required/populated assignment slots: 392/392.
- Unique items: 313.
- Null responses: 0.
- Duplicate AI-pass disagreements/AI consensus records: 33/33.
- All frozen manifest hashes match.
- Protected response hashes remain:
  - `C0C22508E24E7`: `70f28eccc2d3a622c6a4fd322a9dee1f0a148fbe00e01c509488523864835694`.
  - `C0D27B651D266`: `4bab4295da094478d0fad5877366c8d919d2d7960f86cf2c949c15079075ee9d`.

## Main frozen-test results

- Candidate recall: full-pool `1.0`, Recall@10 `0.2308`, Recall@20 `0.4872`.
  Full-pool recall is expected because reference pairs came from candidate pools;
  cutoff recall is the meaningful ranking result.
- Existence-only support: accuracy `0.2316`, macro-F1 `0.0940`, false acceptance
  `1.0`, unsupported rejection `0.0`.
- Semantic NLI support: accuracy `0.3684`, macro-F1 `0.2855`, false acceptance
  `0.0`, unsupported rejection `1.0`. Partial-support F1 remains `0.0`.
- Hypotheses: 70 hypotheses, aggregate mean `3.2419/5`, hypothesis acceptance
  `0.6143`. AI-pass agreement is saved per dimension. Feasibility/model and HNS
  correlations are `insufficient_data` because paired values do not exist.
- Embeddings: MiniLM versus SPECTER NDCG@10 `0.7434` versus `0.3568`, and
  Recall@10 `0.2936` versus `0.1274`, on provisional non-human retrieval
  references.
- LLM contradiction judge: Qwen 27B accuracy/macro-F1 `0.7436/0.4943`; Qwen 8B
  `0.7179/0.4666`; both coverage `1.0`. Both missed the single contradiction,
  giving contradiction F1 `0.0`.

## Controlled-comparison safeguards

- NER changed only the concept extractor on the same seeded 500 documents and
  did not modify the production graph.
- Embeddings changed only the local cached encoder, with identical queries,
  documents, candidate universe, cosine metric, and cutoffs; production Chroma
  was not overwritten.
- LLM comparison changed only the Groq model ID. Prompts were blinded and
  identical, with temperature 0, 1,000 output tokens, reasoning disabled,
  threshold 0.5, and batch size 4.
- Reference annotations, reasons, reviewer identities, adjudication, and
  protected system sidecars were excluded from LLM prompts.

## Verification

- Focused completion suite: 30 passed.
- Complete automated suite: 247 passed.
- Twelve completed packets passed strict packet validation.
- `python -m compileall -q src tests`: passed.
- `python -m pip check`: no broken requirements.
- `git diff --check`: enforced for scoped commits.

## Live integration result

The final fully authorized read-only run completed with 7 passes, 0 failures,
and `all_required_passed: true`:

- Chroma read/count: 8,850 records.
- Neo4j read/count: 8,850 papers.
- Cached local NLI inference: valid supported decision.
- Groq model listing: passed.
- Semantic review: 2/2 retrieved citations verified, 5/5 generated claims
  structurally grounded, and semantic support enabled.
- Contradiction path: one candidate evaluated with a valid `DIFFERENT SCOPE`
  verdict at confidence `0.88`.
- Hypothesis path: one evidence-ranked hypothesis generated with valid `MEDIUM`
  feasibility.

Sanitized evidence is saved under
`results/reasoning/final_ai_reference/live_checks/`; no credentials are stored.

## Limitations and negative findings

- AI references can share model biases and do not establish human validity.
- There is only one contradiction in the frozen test references and none in
  development, so contradiction F1 is unstable and model selection cannot be
  based on that class.
- Semantic NLI improves the existence baseline but performs poorly on partial
  support.
- The fixed NER pattern extractor is conservative and is not a validated
  scientific NER replacement.
- Retrieval relevance judgments are provisional non-human references.
- No statistical significance or large-corpus scaling claim is made.
