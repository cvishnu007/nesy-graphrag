# Evaluation Protocol

Status: adopted November 13, 2026. This defines what the project reports as results, how configurations are selected, and which historical metrics are diagnostics only. Read with [AGENT_HANDOFF.md](../AGENT_HANDOFF.md).

## 1. Principles

1. **Select on dev, confirm once on test.** Every configuration choice (weights, thresholds, gates, prompts, models) is made on development queries. A selection file is written *before* any new variant is run on test, and test is run once per frozen selection.
2. **Every number names its corpus, labels, and judge.** Report the corpus (paper count and clean-file SHA-256), the query set version, the label source (AI reference or human), and the judge model and prompt version.
3. **Complete or explicitly incomplete judgments.** When a system retrieves documents outside the judged pool, either extend the pool with the same blinded judge (`src/evaluation/pool_extension.py`), or report standard and condensed scores together with the unjudged rate. Never report condensed scores alone.
4. **Uncertainty is mandatory.** Report paired bootstrap 95% confidence intervals and exact paired randomization p-values (`src/evaluation/significance.py`). With n queries the smallest attainable two-sided randomization p-value is 2/2^n (n=6 gives 0.031; n=14 gives 0.00012).
5. **AI labels are references, not ground truth.** Results on AI-reference labels are provisional until a human-labeled sample measures judge agreement (Section 5).

## 2. Retrieval (primary task)

| Metric | Definition | Role |
|---|---|---|
| NDCG@10 | Graded gain 2^g − 1, grades 0/1/2, ideal from all judged documents | Primary |
| Recall@20 | Relevant (g ≥ 1) retrieved in top 20 / all judged relevant | Secondary; pool-relative, not corpus-exhaustive |
| P@10, MAP, MRR | Standard binary definitions (g ≥ 1 relevant) | Supporting |
| Unjudged@k | Fraction of top-k without a label | Required whenever scores are reported |

Baselines that must appear in every retrieval table: BM25 (k1 = 1.5, b = 0.75, title + abstract), vector-only (each embedding index used), and the current production configuration.

## 3. Grounded generation (review mode)

The historical `TS` score measures output format compliance, not correctness: any well-formed answer scores 1.0. Report instead:

| Metric | Definition |
|---|---|
| Structural citation validity | Valid cited passage IDs / all cited IDs (what TS's first half measured) |
| Claim support precision | Claims judged SUPPORTED or PARTIALLY_SUPPORTED by their cited passages / claims emitted. Judged by NLI with a human-labeled calibration sample. |
| Abstention rate | Queries with zero accepted claims / queries |
| Claims per answer | Mean accepted claims (prevents a single trivial claim from scoring perfectly) |

Compare vector-RAG and GraphRAG with `src/evaluation/matched_rag.py` (identical generator, prompt, and context budget).

## 4. Contradiction and hypothesis tasks

**Contradiction.** Report candidate recall@k against a benchmark with at least 10 CONTRADICTION positives per split (`support_calibration.coverage_report` checks this), verdict macro-F1 and contradiction-class precision/recall/F1, and output validity (coverage). The current frozen benchmark has 1 positive in 39 test pairs and 0 in dev, so it cannot support a contradiction-quality claim. Candidate ranking signals (concept overlap, year gap) are screening heuristics and are not evidence of disagreement.

**Hypotheses.** Use the 1/3/5 rubric (evidence, novelty, feasibility, specificity, usefulness) from human raters, with quadratic-weighted Cohen's kappa on a double-rated subset, and acceptance under the predeclared rule in `hypothesis_metrics.rating_accepted`. AI ratings may be used only to pilot the rubric.

## 5. Judge reliability

Before publication, two people independently grade a stratified random sample of at least 200 query-document pairs from the frozen pool, blinded to method and AI grade. Report human-human and human-AI weighted kappa and the AI judge's per-grade confusion. If human-AI agreement is weak, rescore with human labels on the sample and state the effect. The current AI judge uses grade 1 rarely (126 of 1,145 labels), so graded NDCG is close to binary; this must be stated.

## 6. Diagnostics, not results

These remain in `src/pipeline/metrics.py` for inspecting runs. They are not reported as quality measures:

| Diagnostic | Why it is not a result |
|---|---|
| TS | Format compliance only (Section 3) |
| NBR | Graph participation, not benefit; replaced by the ablation Δ NDCG@10 with CI |
| ATD | Year spread is driven by the corpus distribution (13,116 papers from 2020, 2,086 from 2025), not relevance |
| RDI | Sums two unrelated counts; no interpretation |
| HNS | Concept-path distance. Before generic-concept flags it was ≈ 0.33 for almost any pair (28 of 29 random pairs had path length 2). Must be validated against human novelty ratings before use. |

## 7. Recorded selections

| Date | Component | Selection | Evidence |
|---|---|---|---|
| 2026-11-13 | Retrieval (BGE era) | `full_bge_bm25_similar`; test 0.619 vs production 0.449, p = 0.004 (confirmed) | `results/retrieval/variant_ablation_bge_dev_20261113/selection.json` |
| 2026-11-14 | Embedding model | BGE-base-en-v1.5 over SPECTER, SPECTER2 base, SciNCL, MiniLM, OpenScholar | `results/external/` |
| 2026-11-14 | Entity extraction | SciBERT fine-tuned on SciERC train, best dev epoch, test once (typed F1 0.639) | `results/external/scierc_ner_comparison_20261114.json` |
| 2026-11-15 | Citation recommendation | RRF(BGE, BM25, co-citation), k = 50 neighbours, weight 1.0, chosen on 200 dev papers, test once on 400 | `results/retrieval/citation_recommendation_20261115/` |
| 2026-11-19 | GraphRAG baseline | LightRAG-style mix and hybrid, LightRAG defaults (top 60), no tuning; pool-extended with the same 7B judge | `results/retrieval/stratified_v2_lightrag_20261119/` |
| 2026-11-19 | Public citation benchmarks | Settings fixed from 2026-11-15 (no tuning on DBLP/PubMed); validation year run first, test years once | `results/retrieval/citeomatic_20261119/` |

Earlier selection:

| Date | Split used | Selection | Evidence |
|---|---|---|---|
| 2026-11-13 | dev (6 queries, pool-extended) | `full_ta_bm25_similar`: SPECTER title+abstract index, BM25 RRF arm (weight 1), graph expansion over CITES and SIMILAR_TO (limit 50) with relative gate (margin 0.03) | `results/retrieval/variant_ablation_dev_20261113/selection.json` |

One-time test outcome (14 queries, pool-extended): selected 0.476 NDCG@10, production 0.449 (p = 0.56), BM25 0.529 (selected − BM25 = −0.053, p = 0.41). Not confirmed; production defaults unchanged. Six dev queries are too few for selection; use the stratified set (22 dev queries) for the next selection round.
