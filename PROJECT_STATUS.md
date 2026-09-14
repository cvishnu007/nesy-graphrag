# NeSy-GraphRAG Project Status

Last updated: September 14, 2026. **Paused; partial implementation retained, laptop runtime additions removed at the user’s request.** Working branch: `feat/local-evaluation`, based on the latest `origin/phase3`. Changes and verification are recorded in [Chinmay.md](Chinmay.md); operational commands are in [docs/LOCAL_EVALUATION.md](docs/LOCAL_EVALUATION.md).

## Current laptop and pause state

The local demo is stopped. Ollama, Neo4j and the session-downloaded generation/NLI models were removed. Original `.env` and 10,000 raw / 8,850 clean / 8,850 NER papers were restored, with an 8,850-paper Chroma index using the original vectors. The pre-existing virtual environment and SPECTER cache remain. See the [cleanup record](results/local/runtime_cleanup_20260914.json).

Retrieval grading stopped at **778 / 1,145 candidates** after losing the Ollama connection; **367 remain**. All development candidates and part of the test pool have labels. No final expanded retrieval scores, significance results, matched-RAG outputs, or live backend-generation checks were produced. The [partial checkpoint](results/retrieval/expanded_20260913/PAUSED.json) and all frozen inputs are retained for a later continuation.

## Current implementation

The abstract-based prototype supports local Ollama generation, SPECTER/Chroma retrieval, Neo4j citation/concept expansion, sentence-level review provenance, structured contradiction decisions, evidence-ranked hypotheses, and Streamlit. Groq remains an explicit optional provider. PDF/full-text ingestion and model training are outside this implementation.

The five follow-up workstreams have produced:

| Workstream | Current evidence |
|---|---|
| Ollama integration | Common completion client and migrated production/general evaluation consumers; real local completion succeeded without Groq credentials. |
| Local demo and matching stores | Dedicated Neo4j plus 47,627-paper corpus, completed NER and Chroma, exact ID and title/abstract parity verified. Streamlit startup passed; backend generation checks remain unfinished. Runtime resources were subsequently removed. |
| Benchmark paths/reproduction | Defaults resolve to the merged `teammate2/` package; offline reproduction verifies hashes and all 392 annotation assignments with zero task failures. |
| Retrieval baselines and refresh | BM25, production-fusion comparison, immutable ranking pools, checkpointed AI judgments and matched-budget RAG implemented. Expanded rankings frozen for all 20 queries; grading paused at 778/1,145, held-out scoring and matched-RAG not run. |
| Reasoning evaluation | Dev-only partial-support calibration and a separate published SciFact claim/evidence benchmark evaluated; improvements and failure tradeoffs recorded below. |

## Verified September 13 snapshot — subsequently removed locally

The user did not have the teammate's expanded artifacts. This Mac regenerated the corpus from the existing 8,850-paper dataset and resumed the 11-topic Semantic Scholar download. Counts differ slightly from the teammate's August 31 snapshot because the upstream fetch differs.

| Artifact | September 13 count |
|---|---:|
| Unique raw papers | 52,820 |
| Clean papers | 47,627 |
| Completed NER papers | 47,627 |
| Chroma `s2_papers` vectors | 47,627 |
| Neo4j Paper nodes | 47,627 |
| Neo4j Author nodes | 146,157 |
| Neo4j Concept nodes | 196,616 |
| AUTHORED_BY relationships | 190,284 |
| RELATED_TO relationships | 475,199 |
| Real CITES relationships | 22,514 |

All original 10,000 raw and 8,850 clean IDs were retained. The [store audit](results/local/store_audit_20260913.json) found zero missing IDs, extra IDs, duplicate IDs, unfinished NER records, or title/abstract mismatches across clean/NER/Chroma/Neo4j. It does not independently re-encode each stored vector.

Local runtime: Python 3.12.13, an 8 GB Apple Mac using MPS for SPECTER and two CPU workers for NER, Ollama 0.33.3 with `qwen2.5:3b-instruct`, Neo4j 2026.07.1. Neo4j runs in a dedicated loopback instance; credentials and stores are ignored by Git. Its reset now deletes in 1,000-node transactions after the prior whole-graph transaction exceeded the configured memory limit.

The teammate's earlier **47,619 clean papers / 22,370 citation edges** remain historical evidence in `VISHNU.md` and the prior Git revision of this file. Imported teammate live-check outputs used an older **8,850-paper** store. Do not mix those snapshots with this run.

## Retrieval evaluation

The original 20 query texts and 6 development / 14 test assignments remain frozen. All queries concern graph-neural-network topics; they do not measure the full breadth or balance of the 11-topic corpus.

The expanded pool contains **1,145 distinct query-paper candidates** from five methods, each retrieved to depth 20: vector-only, graph-only, the historical evaluation hybrid (16:1 vector/graph weights), BM25 (k1=1.5, b=0.75), and production graph-filtered fusion. Settings and ranks were captured before new judgments, with artifact hashes preventing accidental modification. BM25 uses title plus abstract; historical SPECTER indexing uses abstracts.

Judgments are local Ollama AI references from method-hidden title/abstract evidence, with raw responses, checkpoint identity checks, and a CSV for human review. They are not human ground truth. Recall is relative to the pooled judgments, not exhaustively known relevance across 47,627 papers. Cached-ranking evaluation timings are not live retrieval latency.

The controlled RAG comparison uses the same generator, prompt, temperature 0, top 5 papers, 6,000 formatted Unicode characters of evidence, 600 output tokens, and zero repairs for vector RAG and GraphRAG. Both validate IDs against the clean corpus. Evidence and raw outputs remain available for audit. Passage-ID validity measures traceability; it does not prove semantic correctness or answer quality.

Expanded retrieval scores and matched-RAG output statistics are unavailable because the runs are incomplete and paused. No retrieval settings were tuned on the new test judgments.

Historical 8,850-paper test results are retained only for context: evaluation hybrid NDCG@10 **0.3676**, vector-only **0.3643**, paired randomization **p=1.0**. They did not establish a significant hybrid advantage and are not expanded-corpus results.

## Reasoning evaluation

The frozen package contains 62 contradiction items (23 dev / 39 test), 151 claim-support items (56 / 95), and 100 hypothesis items (30 / 70). It has 313 unique annotation items and 392 reviewer assignments. Repeated AI annotation passes produced these references; no new human validation is claimed. [Offline verification](results/reasoning/package_verified_20260913/verification.json) checks the frozen hashes and reports older package-manifest drift separately.

Development-only calibration of saved NLI probabilities produced these results on the disjoint historical AI-reference test set:

| Metric | Original floor 0.20 | Dev-selected floor |
|---|---:|---:|
| Partial-support F1 | 0.0000 | 0.1250 |
| Four-class macro F1 | 0.2855 | 0.3190 |
| False acceptance rate | 0.0000 | 0.03125 |

This is a post-hoc baseline experiment, not an independent replication or a production-model improvement. Prior test weakness motivated it, so a fresh untouched test set is needed for confirmatory claims. The chosen threshold was saved before opening test predictions and has **not been adopted in production**. [Selection and metrics](results/reasoning/support_calibration_20260913/test_metrics.json) retain the tradeoff.

The separate [SciFact rationale benchmark](evaluation/benchmarks/scifact_support_v1.json) imports published expert labels, preserves complete rationale sets, and excludes duplicate content across partitions. Official train is local development (955 items); official dev is local test (338 items: 216 supported, 122 contradicted). Official test labels are unavailable. Source claims/labels are CC BY 4.0 and abstracts ODC-By 1.0.

Fixed `cross-encoder/nli-deberta-v3-small`, pinned revision `fa2804872c3b4bd748f38c0185cc85775361e735`, floor 0.20, with no SciFact test tuning:

| Metric | SciFact local test |
|---|---:|
| Overall accuracy | 0.3580 |
| Contradiction precision | 0.7500 |
| Contradiction recall | 0.5410 |
| Contradiction F1 | 0.6286 |
| Supported F1 | 0.4000 |
| Two-gold-class macro F1 | 0.5143 |

All four prediction labels remain in the confusion matrix; macro F1 averages only the two gold classes. This biomedical claim/evidence task **does not validate graph candidate discovery or paper-to-paper contradiction** and has no partial/unsupported gold classes. The old in-domain frozen contradiction set still has one positive test contradiction and zero development positives. Its class coverage remains inadequate for strong conclusions.

## Verification and remaining limits

- After laptop cleanup: **306 tests passed**, one upstream spaCy/Click deprecation warning. Original 8,850-ID clean/NER/Chroma parity and closed service ports were also verified.
- `src`, `app`, and tests compile; `pip check` reports no broken requirements.
- Store identity/evidence-text audit passed; provider model metadata and source hashes are saved.
- The independent review agent hit its service usage limit; code review continued locally.
- Streamlit startup AppTest found no exceptions on September 13. The server is now stopped and uninstalled dependencies must be restored before using the demo. Pipeline-generation checks remain unfinished.

Remaining research work is independent review of retrieval judgments and generated answers, a better-covered in-domain paper-pair contradiction benchmark, expert hypothesis evaluation, and semantic-support validation that addresses false acceptance. Broader model/component comparisons and controlled scaling tests should use fixed development/test protocols. The corpus is not balanced across all computer science, the graph is sparse, generic noun chunks are noisy, and a single operational rebuild is not a scaling benchmark.
