# Results (November 13–15, 2026)

Every number below states its corpus, label source, and statistical test. Protocol: [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md). Raw artifacts are in `results/`.

## 1. Public benchmarks with external labels

Labels come only from the benchmarks. Paired Monte Carlo sign-flip tests (20,000 permutations) with bootstrap 95% CIs (`src/evaluation/external_benchmarks.py`).

**SciDocs** (BEIR; 1,000 queries, 25,657 papers; citation/co-view-derived labels). Published reference values reproduced: MiniLM 0.216 (ours 0.217), BM25 0.158 (ours 0.152).

| Method | NDCG@10 | Recall@100 | Δ vs SPECTER |
|---|---:|---:|---|
| BGE-base-en-v1.5 | **0.218** | 0.493 | +0.068, p < 0.001 |
| MiniLM-L6-v2 | 0.217 | **0.510** | +0.066, p < 0.001 |
| BM25 + SciNCL (RRF) | 0.199 | 0.479 | +0.048, p < 0.001 |
| SciNCL | 0.196 | 0.508 | +0.045, p < 0.001 |
| BM25 + BGE (RRF) | 0.198 | 0.468 | +0.047, p < 0.001 |
| SPECTER2 base (no adapter) | 0.173 | 0.463 | +0.022, p < 0.001 |
| OpenScholar retriever | 0.163 | 0.408 | +0.012, p = 0.026 |
| BM25 | 0.152 | 0.350 | n.s. |
| SPECTER (historical production) | 0.151 | 0.413 | — |

**LitSearch** (Ajith et al., 2024; 597 literature-search questions, 64,183 ACL/ICLR papers; author-inspected targets).

| Method | NDCG@10 | Recall@5 | Recall@20 | Δ vs SPECTER |
|---|---:|---:|---:|---|
| **BM25 + BGE (RRF)** | **0.490** | **0.547** | **0.685** | +0.281, p < 0.001 |
| BM25 + SPECTER2 base | 0.462 | 0.524 | 0.662 | +0.254 |
| BGE | 0.449 | 0.506 | 0.669 | +0.240 |
| BM25 + OpenScholar | 0.410 | 0.480 | 0.598 | +0.202 |
| BM25 | 0.396 | 0.445 | 0.602 | +0.187 |
| SPECTER2 base | 0.331 | 0.388 | 0.560 | +0.123 |
| OpenScholar | 0.277 | 0.330 | 0.477 | +0.069 |
| SciNCL | 0.241 | 0.269 | 0.427 | +0.033 |
| SPECTER | 0.208 | 0.244 | 0.371 | — |

OpenScholar vs BGE: −0.055 (SciDocs) and −0.172 (LitSearch), both p < 0.001. Findings: (1) every alternative beats SPECTER; (2) BM25 fusion helps every dense model on LitSearch; (3) a strong general retriever beats all scientific embedding models tested, and SciNCL's SciDocs strength does not transfer to natural-language questions. SPECTER2 was evaluated without its query adapter (the `adapters` package would downgrade `transformers`), so its scores are a lower bound.

## 2. Project benchmark (20 GNN queries; AI-reference labels; pool-extended) — superseded by §2c

> **Superseded.** These numbers use the frozen 3B labels, which agree with human grades at κ 0.17 (§2b). They are kept for the record. Use §2c (same rankings, validated labels) and §2d (independent 44-query set).

Frozen pool: 1,145 labels from local `qwen2.5:3b-instruct` (grades 0/1/2: 541/126/478), on the 47,627-paper pool corpus. New variants run on the 47,619-paper live stores; their unjudged results were graded by the same blinded judge (`pool_extension.py`), so every scored list is fully judged. Exact paired randomization tests.

| Test (14 queries) | NDCG@10 | P@10 | Δ vs production | p | Wins |
|---|---:|---:|---:|---:|---:|
| **Selected: BGE + BM25 + graph (CITES + SIMILAR_TO, relative gate)** | **0.619** | 0.707 | +0.171 | **0.004** | 12/14 |
| BGE alone | 0.620 | 0.721 | +0.171 | 0.0005 | 13/14 |
| BGE + BM25 | 0.566 | 0.664 | +0.117 | 0.024 | 10/14 |
| BM25 | 0.529 | 0.629 | +0.080 | — | — |
| SPECTER-era best (title+abstract + BM25 + graph) | 0.475 | 0.600 | +0.027 | 0.56 | 7/14 |
| Production before Nov 13 (SPECTER, absolute gate) | 0.449 | 0.507 | — | — | — |

The selection was frozen on dev (6 queries) before test was opened (`results/retrieval/variant_ablation_bge_dev_20261113/selection.json`). Non-selected BGE + graph without BM25 scored 0.646 on test; it is reported but was not chosen post hoc. The earlier SPECTER-era dev selection did not generalize (dev 0.548 → test 0.476 < BM25 0.529).

## 2b. Relevance judge validation against human grades

One annotator graded a deterministic 200-pair sample of the pool, blinded to the method and the AI grade (`evaluation/human_review/relevance_sheet_cv.csv`). The pairs were split 50/50 by hash. Judges were compared on the selection half, and the chosen judge was scored once on the held-out half (`src/evaluation/judge_agreement.py`; `results/retrieval/judge_agreement_*_20261114/report.json`). Agreement is quadratic-weighted Cohen's kappa on the 0/1/2 scale.

| Judge | Selection κ (n=100) | Held-out κ (n=100) | Held-out exact | Held-out binary |
|---|---:|---:|---:|---:|
| Frozen pool labels (qwen2.5-3B, pool batches) | 0.166 | — | — | — |
| qwen2.5-3B, original prompt, re-run | 0.563 | 0.441 | 0.53 | 0.62 |
| qwen2.5-3B, strict prompt | 0.329 | — | — | — |
| qwen2.5-7B (local), original prompt, 4 papers per call | 0.581 | — | — | — |
| qwen2.5-7B (local), strict prompt, 4 papers per call | 0.602 | — | — | — |
| **qwen2.5-7B (local), strict prompt, 1 paper per call (used for grading)** | **0.769** | **0.749** | **0.72** | **0.81** |
| Qwen3.8-27B (Groq), original prompt | 0.817 | — | — | — |
| **Qwen3.8-27B (Groq), strict prompt (best)** | **0.883** | **0.884** | **0.81** | **0.85** |

The 27B judge is the most accurate, but Groq's free tier (200K tokens/day) covers only about 200 pairs a day, against about 3,950 needed. The local 7B judge, one paper per call, passes the predeclared threshold (held-out κ ≥ 0.6) and grades both pools. Grading one paper per call raised the 7B's κ from 0.60 to 0.77, consistent with the batch-position bias below.

All 200 frozen labels against the human grades: κ 0.187. Diagnosis: the stored labels match the raw judge outputs exactly (0 mismatches of 1,145), so the problem is not storage or alignment. The 3B judge has a batch-position bias. Across 279 four-paper batches it used only 20 of the 81 possible grade patterns, gave position 2 a grade of 2 in almost every batch, and produced `[0,2,2,0]` or `[0,2,0,2]` in 40% of them. Re-running the same model on the same pairs agreed with the frozen labels at only κ 0.031. **The frozen 3B labels are therefore treated as unreliable, and section 2 is superseded by section 2c.** Limitation: one annotator, so human-human agreement is not measured.

## 2c. GNN benchmark regraded with the validated judge

Every pooled pair (1,145 frozen plus 499 extension; 1,644 total) was regraded by the local 7B judge (strict prompt, one paper per call, held-out κ 0.75), papers ordered by hash within each query (`src/evaluation/regrade_pool.py`; `results/retrieval/regraded_qwen2_5_7b_strict_b1_20261114/summary.json`). Rankings are the stored Nov 13 rankings, unchanged; every scored list is fully judged. Exact paired randomization tests.

| Test (14 queries) | NDCG@10 (validated labels) | NDCG@10 (frozen 3B labels, §2) |
|---|---:|---:|
| **Selected: BGE + BM25 + graph** | **0.813** | 0.619 |
| BGE alone | 0.814 | 0.620 |
| BGE + graph | 0.813 | 0.646 |
| BM25 | 0.793 | 0.529 |
| BGE + BM25 | 0.791 | 0.566 |
| Production before Nov 13 (SPECTER, absolute gate) | 0.349 | 0.449 |

Selected vs production: +0.464, p = 0.0001, 14/14 wins. Selected vs BGE alone: −0.001, p = 0.96. Selected vs BM25: +0.020, p = 0.67. On dev (6 queries), the selected configuration ranks first under the validated labels too (0.701), so the Nov 13 selection stands. Conclusion: BGE-family retrieval is far better than the SPECTER-era system; on GNN queries, BM25 and graph expansion add nothing measurable over BGE.

## 2d. Independent confirmation: 44 topic-stratified queries

Queries: `evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json` v0.5 (4 per ingestion topic, 11 topics; team-reviewed, frozen before pooling; two near-empty queries replaced after a corpus-support check). The retrieval configuration was frozen on Nov 13 and nothing was tuned on these queries. Pool: union of the top 20 of 8 systems, 2,305 pairs, all graded by the validated 7B judge (`src/evaluation/stratified_benchmark.py`; `results/retrieval/stratified_v2_20261114/summary.json`). Monte Carlo sign-flip tests (20,000) with bootstrap 95% CIs; 9 predeclared comparisons.

| All 44 queries | NDCG@10 | P@10 | Recall@20 |
|---|---:|---:|---:|
| **Selected: BGE + BM25 + graph (CITES + SIMILAR_TO)** | **0.644** | **0.725** | **0.613** |
| BM25 | 0.626 | 0.691 | 0.534 |
| BGE + BM25 | 0.621 | 0.702 | 0.588 |
| BGE alone | 0.593 | 0.684 | 0.556 |
| BGE + graph | 0.585 | 0.680 | 0.589 |
| SPECTER-era best (title+abstract + BM25 + graph) | 0.380 | 0.486 | 0.405 |
| Production before Nov 13 (SPECTER, absolute gate) | 0.287 | 0.382 | 0.264 |
| SPECTER vector only | 0.273 | 0.368 | 0.263 |

| Paired comparison (44 queries) | Δ NDCG@10 | 95% CI | p | Holm-adjusted |
|---|---:|---|---:|---|
| Selected vs production | +0.358 | [0.285, 0.434] | < 0.0001 | significant |
| BM25 vs production | +0.339 | [0.270, 0.406] | < 0.0001 | significant |
| BGE + BM25 vs production | +0.334 | [0.265, 0.405] | < 0.0001 | significant |
| BGE vs production | +0.306 | [0.235, 0.381] | < 0.0001 | significant |
| Selected vs BGE alone | +0.052 | [0.012, 0.087] | 0.010 | significant (threshold 0.010) |
| Selected vs BGE + BM25 (graph contribution) | +0.023 | [−0.011, 0.059] | 0.21 | n.s. |
| Selected vs BM25 | +0.019 | [−0.029, 0.064] | 0.46 | n.s. |
| BGE vs BM25 | −0.033 | [−0.090, 0.024] | 0.27 | n.s. |
| BGE + graph vs BGE | −0.008 | [−0.053, 0.034] | 0.74 | n.s. |

Dev and test halves (22 queries each) agree: selected 0.710 / 0.578; production 0.306 / 0.268. Two operating-systems queries have no grade-2 paper in the pool (V2-11-2, V2-11-3).

**Findings.** (1) The selected system beats the original production system by a large margin on every topic split. (2) Its advantage over BGE alone comes from BM25 fusion. Graph expansion over BGE + BM25 is a small, non-significant gain (+0.023), and graph expansion alone does not help BGE (−0.008). (3) BM25 is a strong baseline on this corpus; the selected system's lead over it is not significant. Retrieval claims must therefore credit lexical fusion, not the graph.

## 2e. Retrieval on the densified citation graph

`src/evaluation/dense_graph_eval.py`; `results/retrieval/dense_graph_20261115/`. The selected system's graph expansion was rerun with the opt-in dense layers (§6). Newly retrieved pairs (67 dev, 48 test) were graded by the same validated 7B judge into a separate extension file. The rule was predeclared: pick the best dense variant on the 22 dev queries and test it once.

| NDCG@10 | Dev (22) | Test (22) |
|---|---:|---:|
| Current system (`CITES` + `SIMILAR_TO`) | 0.708 | 0.576 |
| BGE + BM25, no graph | 0.695 | 0.543 |
| **Dense: `CITES` + `CITES_ADDED` + `COUPLED_WITH` (selected on dev)** | 0.693 | 0.540 |
| Dense: all four edge types | 0.680 | — |
| Dense: `CITES` + `CITES_ADDED` + `SIMILAR_TO` | 0.657 | — |

On test, dense vs no graph: −0.003 (p = 0.92); dense vs current: −0.036 (p = 0.20). Recall@20 is slightly higher with the dense graph than without a graph (0.605 vs 0.585).

**Why the denser graph does not raise NDCG.** On dev, the 63 papers that dense expansion moved into the top 10 have the same mean relevance as the 63 they displaced (1.19 vs 1.21 on 0–2; 51% grade 2 in both). The graph finds equally relevant alternatives, but BGE + BM25 already fills the top 10 with relevant papers (P@10 ≈ 0.78), so swapping cannot raise the score. Topical-relevance ranking for short queries is the setting where citation structure helps least. Its measurable value in this project is elsewhere: citation-grounded contradiction pairs (§5b-2), hypothesis candidates outside the vector neighbourhood (§4b), and coverage (§6).

## 2f. Citation recommendation: where the graph helps

Task: given a corpus paper's title and abstract, find the corpus papers it cites (`src/evaluation/citation_recommendation.py`; `results/retrieval/citation_recommendation_20261115/`). Ground truth is the real reference lists (`CITES` + `CITES_ADDED`), with no LLM judge.

**Leakage controls:**
- the query paper's own citation edges are never used;
- papers that cite it are excluded from the neighbour set;
- every method ranks only papers published no later than the query paper.

The graph signal is co-citation through neighbours: papers cited by the query's 50 nearest BGE neighbours, weighted by neighbour similarity. Of 9,534 eligible papers (year ≥ 2022, ≥ 5 older cited corpus papers), 600 were selected by hash: 200 dev, 400 test. Graph settings were chosen on dev by Recall@20 from a predeclared grid (k ∈ {10, 20, 50}, weight ∈ {0.5, 1.0}). The chosen setting, k = 50 with weight 1.0, is at the edge of the grid and ran once on test. Monte Carlo sign-flip tests with bootstrap 95% CIs.

| Test (400 papers) | Recall@20 | NDCG@10 | MAP |
|---|---:|---:|---:|
| **BGE + BM25 + co-citation graph** | **0.308** | **0.283** | **0.188** |
| Co-citation graph alone | 0.267 | 0.250 | 0.159 |
| BGE | 0.251 | 0.231 | 0.142 |
| BGE + BM25 | 0.241 | 0.217 | 0.135 |
| BM25 | 0.169 | 0.155 | 0.091 |

| Comparison (test) | Δ Recall@20 | 95% CI | Δ NDCG@10 | p |
|---|---:|---|---:|---:|
| With graph vs BGE + BM25 | **+0.068** | [0.055, 0.081] | +0.067 | < 0.0001 |
| With graph vs BGE | **+0.057** | [0.043, 0.072] | +0.053 | < 0.0001 |
| Graph alone vs BGE | +0.016 | [−0.006, 0.038] | +0.019 | 0.17 |

Dev agrees (+0.071 Recall@20 over BGE + BM25). **Finding:** when the task depends on citation structure rather than topical similarity, the densified graph gives a large, significant gain (28% relative Recall@20 over BGE + BM25). Together with §2d–2e, this marks the boundary: graph structure is redundant for short topical queries but complementary for citation-level relatedness. BM25 fusion slightly hurts here (long abstract queries), unlike in §2d.

## 3. Scientific entity extraction (SciERC test; 100 abstracts, 1,685 gold entities)

| Extractor | Typed F1 | Typed F1 w/o Generic | Span F1 |
|---|---:|---:|---:|
| **SciBERT fine-tuned on SciERC (this project)** | **0.639** | **0.624** | **0.747** |
| GLiNER medium v2.1, zero-shot | 0.240 | 0.266 | 0.378 |
| Community SciERC SciBERT (anlp) | 0.294 | 0.187 | 0.297 |
| Community SciERC SciDeBERTa (anlp) | 0.242 | — | 0.255 |
| Community SciERC SciBERT (roccas98) | 0.128 | — | 0.151 |
| spaCy noun chunks + entities (historical) | — | — | 0.270 |

Fine-tuning: `allenai/scibert_scivocab_uncased`, 10 epochs, lr 5e-5, best dev epoch 9 (dev F1 0.661), test scored once; 264 s on an RTX 3050. Published SciBERT SciERC NER is about 0.65–0.67, so this is in range. Community checkpoints shipped without tokenizers; with the correct base tokenizer they remain far below a properly trained model.

Corpus extraction (47,619 abstracts): 1,225,433 typed mentions (Method 379k, OtherScientificTerm 376k, Task 281k, Material 115k, Metric 74k), 323,624 `SciEntity` nodes after normalization, 696,390 `MENTIONS` edges (≤ 15 per paper).

## 4. Concept graph comparison (26 probe queries, live stores)

| | spaCy raw | spaCy, generic filtered | **Scientific entities** |
|---|---:|---:|---:|
| Hypothesis candidates / query | 25.0 | 22.6 | 25.0 |
| Concept-overlap contradiction pairs / query | 4.96 | 0.92 | 0.85 |
| HNS values on random pairs | {0.333} only | {0.333, 0.667, 1.0} | {0.333, 0.667} |

Representative shared concepts behind the top hypothesis candidate: "molecular property prediction", "drug discovery" (vs "graph neural networks", "algorithms"); "malicious software", "malware" (vs "them", "this report"); "text-to-sql", "sql queries" (vs "resources", "input"). Concept-overlap contradiction pairing is sparse once noise is removed, so contradiction candidates use NLI sentence pairs.

## 4b. Hypothesis candidates: human ratings (round 1, one rater) and revision

30 stratified queries; per query, the top structural-hole candidate from the entity graph and a vector-neighbour baseline (vector rank 6–15, not cited by the top 5), shuffled and blinded (`evaluation/human_review/hypothesis_sheet_rated_cv.csv`). Ratings 1–5; paired by query, Monte Carlo sign-flip test.

| Mean rating | Structural hole (graph) | Vector neighbour | Graph − vector | p |
|---|---:|---:|---:|---:|
| Relevance | 3.27 | **4.20** | −0.93 [−1.43, −0.47] | **0.002** |
| Novelty | 1.97 | 2.07 | −0.10 | 0.76 |
| Promise | 2.20 | 2.57 | −0.37 | 0.17 |

**Configuration caveat.** The round-1 structural-hole candidates were generated with `CONCEPT_EXCLUDE_GENERIC=false` (the code default), not the `true` recorded for the application on Nov 14. With the generic filter on, 21 of the 30 top candidates change. Round 1 therefore rates the unfiltered variant. Round 2 uses the filtered one (`build_round2_kit.py` refuses to run otherwise).

The original structural-hole candidates are significantly less relevant and no more novel. The rater's notes ("unrelated: IDS survey", "generic gnn tutorial") trace this to links through broad shared concepts.

**Revision: problem-bridge, novel-method candidates** (`src/pipeline/hypothesis_bridge.py`), following the literature-based-discovery pattern "apply method M to problem T". The candidate shares a specific Task or Material entity (IDF-weighted) with the query papers, mentions a Method that none of the query's 15 nearest papers mention, stays above a relevance floor, and is outside the vector top 15. Development on the same 30 queries used the human-validated 7B relevance judge; the selection rule was predeclared (`src/evaluation/hypothesis_dev.py`):

| Generator | Mean judged relevance (0–2; no candidate = 0) | Coverage |
|---|---:|---:|
| Vector neighbour | 0.93 | 30/30 |
| **Problem bridge, margin 0.10 (selected)** | **0.77** | 29/30 |
| Problem bridge, margin 0.10, specific entities only | 0.77 (tie) | 27/30 |
| Problem bridge, margin 0.05 | 0.70 | 23/30 |
| Original structural hole | 0.43 | 30/30 |

The judge reproduces the human ordering of the two rated arms (0.43 vs 0.93; humans 3.27 vs 4.20). The tie was broken by the implementation's alphabetical order, not by a stated rule, so both tied variants go to the human test. Novelty cannot be judged automatically: round 2 rates all four arms on the 14 queries not used here.

## 4c. Time-sliced hypothesis benchmark (retrospective discovery)

`src/evaluation/temporal_hypothesis.py`; `results/reasoning/temporal_hypothesis_20261115/`. This is the standard literature-based-discovery evaluation, with no human labels.

**Setup.**
- **Hypothesis:** "method M could be applied to task T", for a SciERC (Task, Method) pair that never co-occurs in any paper up to the cutoff.
- **Confirmed:** the pair co-occurs in ≥ 2 papers after the cutoff.
- **What scorers see:** only pre-cutoff papers (32,039; cutoff 2022) with their entities, BGE vectors, and citations between pre-cutoff papers. Entity frequencies and the generic filter (document frequency ≤ 1%, specific names only) are recomputed on pre-cutoff papers.
- **Size:** 120 tasks with ≥ 10 pre-cutoff papers and ≥ 3 confirmed methods; on average 4,633 candidate methods and 5.6 confirmed per task. Scorers were predeclared and nothing was tuned. Paired sign-flip tests across tasks.

| Scorer (120 tasks) | AP | P@20 | Recall@100 | AUROC |
|---|---:|---:|---:|---:|
| **Fusion: popularity + vector + entity graph + citation graph** | **0.087** | **0.050** | **0.441** | **0.927** |
| Fusion: vector + entity graph + citation graph | 0.083 | 0.045 | 0.406 | 0.894 |
| Fusion: entity graph + citation graph | 0.077 | 0.046 | 0.390 | 0.864 |
| Entity-graph bridge (Adamic–Adar) | 0.072 | 0.043 | 0.398 | 0.860 |
| Vector similarity (text only) | 0.044 | 0.033 | 0.302 | 0.846 |
| Popularity | 0.032 | 0.021 | 0.288 | 0.896 |
| Citation-graph bridge | 0.017 | 0.013 | 0.231 | 0.679 |
| Random | 0.003 | 0.003 | 0.024 | 0.506 |

| Comparison | Δ AP | 95% CI | Δ Recall@100 | p (AP) |
|---|---:|---|---:|---:|
| Vector + graph vs vector | +0.039 | [0.022, 0.056] | +0.103 | < 0.0001 |
| Graph only vs vector | +0.033 | [0.015, 0.052] | +0.088 | 0.0004 |
| Entity bridge vs vector | +0.028 | [0.010, 0.047] | +0.096 | 0.001 |
| All four vs popularity | +0.055 | [0.038, 0.073] | +0.152 | < 0.0001 |

**Robustness.** Test half (60 tasks): vector + graph vs vector +0.043 AP (p = 0.003). Cutoff 2021 (135 tasks): 0.142 vs 0.097 (+0.045, p < 0.0001). Stricter confirmation, ≥ 3 later papers (32 tasks): 0.079 vs 0.043 (+0.036, p = 0.035).

**Finding.** Graph structure predicts which method–task combinations researchers adopt later. Graph signals nearly double average precision over text similarity, and the best fusion ranks confirmed combinations about 29× above random. The citation bridge is weak alone but complements the entity graph. This evaluates the signals behind the hypothesis generator, not the generated text; the human round 2 judges candidate quality.

**Limits.** "Confirmed" means co-occurrence in later abstracts. It does not mean a positive result, and it misses combinations described with other names. Entities come from an automatic extractor (typed F1 0.64).

## 4d. Time-sliced test of the hypothesis generators

`src/evaluation/temporal_generator.py`; `results/reasoning/temporal_generator_20261115/`. The §4c setup (cutoff 2022, ≥ 2 confirming papers, same 120 tasks, same candidates) is reused, with each arm proposing 10 methods per task from pre-cutoff data only. The task name is the query.

| Proposes 10 methods per task | Precision@10 | Tasks with ≥ 1 confirmed |
|---|---:|---:|
| **Method–task ranker: RRF(vector, entity bridge, citation bridge), the §4c scorer** | **0.067** | — |
| Entity bridge alone (§4c) | 0.061 | — |
| Vector-neighbour papers' methods (human-study baseline) | 0.038 | 34 |
| Popularity | 0.023 | 26 |
| Paper-candidate bridge generator (`hypothesis_bridge`, pre-cutoff port) | 0.021 | 16 |
| Random | 0.001–0.003 | 1 |

The ranker rows come from the §4c scorers (predeclared; precision@10 added to the same runs). Bridge vs vector neighbour: −0.017, 95% CI [−0.033, −0.001], p = 0.053. Ranker (vector + graph) vs vector scorer: +0.030 [0.017, 0.043], p = 0.0001. The two evaluations agree where they overlap (popularity 0.023 in both; vector 0.037 vs 0.038).

**Finding.** The graph signal is strong, but routing it through candidate papers wastes it. The bridge generator proposes only methods that none of the 15 nearest papers mention, which discards the methods researchers most often adopt next. Ranking (task, method) pairs directly is 1.8× the text baseline and 3× the paper-candidate generator.

**Application.** `src/pipeline/hypothesis_ranker.py` implements the ranker on the full corpus (`HYPOTHESIS_GENERATOR=ranker`). For readability it filters generic heads ("X methods") and task acronyms of 3 characters or fewer. With that filter, the benchmark advantage holds (94 tasks: 0.052 vs 0.028, p < 0.0001; `summary_informative_only.json`). Two caveats: "never combined in the corpus" does not mean novel in the wider literature (the corpus covers 2020–2025), and entity typing errors pass through (e.g., "social engineering" tagged as a method). The ranker has not been human-rated; round 2 rates the paper-candidate generators.

## 5. Contradiction candidates

| Generator | Queries with candidates (of 26) | Evidence |
|---|---:|---|
| Concept overlap, spaCy raw | 23 | none; pairs driven by "this paper", "graph neural networks" |
| Concept overlap, noise removed | 0 (Nov 13 probe) to 10 | none |
| **NLI sentence pairs, author findings only** | **18** | two sentences with passage IDs |

Structured JSON verdicts: 5/5 valid in the end-to-end smoke test (legacy line format: 17 of 39 malformed on the frozen test set). The local 3B judge reports confidence 1.0 for every verdict, so confidence is not calibrated; contradiction accuracy is unmeasured until a benchmark with ≥ 10 positives per split exists.

## 5b. Contradiction verdict models (SciFact test; 338 expert-labelled pairs, 122 contradictions)

Same structured prompt as the application, no tuning, single run (`src/evaluation/contradiction_verdict_eval.py`; `results/reasoning/contradiction_verdicts_summary_20261114.json`).

| Verdict model | Precision | Recall | F1 | Accuracy | Calibration error (ECE) | "Contradiction" calls |
|---|---:|---:|---:|---:|---:|---:|
| qwen2.5-3B-instruct (local) | 0.386 | 0.836 | 0.528 | 0.462 | 0.538 | 264 / 338 |
| qwen2.5-7B-instruct (local) | 0.849 | 0.598 | 0.702 | 0.817 | — | 86 / 338 |
| Qwen3.8-27B (Groq) | 0.954 | 0.672 | 0.788 | 0.870 | 0.089 | 86 / 338 |
| **GPT-OSS-120B (Groq)** | 0.946 | 0.713 | **0.813** | **0.881** | **0.066** | 92 / 337 |
| NLI screening alone (dev threshold) | 0.614 | 0.730 | 0.667 | 0.737 | AUROC 0.818 | — |

GPT-OSS-120B answered 337 of 338 items before Groq's daily token limit (200K TPD); the missing item is a non-contradiction, and the worst-case F1 is 0.809. The local 3B model over-detects contradictions (accuracy below the 0.64 majority baseline) and its confidence is uninformative. Recommendation: NLI screening locally, verdicts from a large model.

## 5b-2. Contradictions in the CS corpus: human labels (round 1, one rater)

68 pairs from the NLI sentence-pair screener (30 stratified queries, top 20 retrieved papers each; `evaluation/human_review/contradiction_sheet_labeled_cv.csv`). Labels: 2 = contradiction, 1 = tension, 0 = none.

| | Pairs | Human contradiction (2) | Human tension (1) |
|---|---:|---:|---:|
| Top-ranked NLI candidates | 56 | 0 | 5 (8.9%) |
| Lower-ranked controls | 12 | 0 | 0 |

The screener mostly pairs papers that each report their own good results. The local 7B verdict model, given the same pairs and the application prompt (`src/evaluation/contradiction_human_eval.py`), called 5 of 68 a contradiction: 2 were human tension pairs, and 3 of the 63 human-0 pairs were false alarms (4.8%). Without the verdict step, all 68 would be shown.

**Revision: citation-grounded index** (`src/pipeline/contradiction_citation.py`, `src/evaluation/contradiction_index.py`). Disagreements in CS are usually written on purpose (replications, "revisiting", negative results), and the disputing paper cites the disputed one. The index pairs critical findings ("fails to", "contrary to", "surprisingly", "on par with") with the papers they cite, requires a shared specific SciERC entity in both sentences, and ranks pairs by NLI. On the densified graph: 364 critical-finding papers, 1,958 citation-linked pairs, 591 candidates. The 7B verdict model called 9 of the top 150 contradictions. Human precision is pending (round 2, 40 pairs).

## 5c. Answer grounding: standard RAG vs this system (matched budget)

14 test queries; identical generator (local qwen2.5-3B), prompt, top-5 papers, 6,000-character evidence budget, zero repair attempts. Claim support judged by local NLI (`cross-encoder/nli-deberta-v3-small`, pinned revision) against the cited passages (`src/evaluation/rag_answer_quality.py`).

| | Standard RAG (BGE vector) | This system (BGE + BM25 + graph) |
|---|---:|---:|
| Claims per answer | 5.79 | 5.93 |
| Structural citation precision | 0.984 | 0.995 |
| **Claims supported by cited evidence** | **0.321** | **0.277** |
| Claims contradicted by cited evidence | 0.062 | 0.120 |

Per-query support difference −0.043, 95% CI [−0.248, +0.150], exact paired p = 0.69: no significant difference. Retrieved papers overlap by about 2.5 of 5 per query. Main finding: structurally valid citations (≈99%) do not imply support (≈30%); passage-ID validation alone overstates grounding. NLI support is an automated proxy pending human validation.

### 5c-2. Answer grounding with the 7B generator (44 stratified queries)

Same protocol as §5c (top 5 papers, 6,000-character budget, zero repair attempts, local NLI support), generator `qwen2.5:7b-instruct`, on all 44 stratified queries (`results/reasoning/matched_rag_7b_stratified_{dev,test}_20261115/`).

| | Standard RAG (BGE vector) | This system (BGE + BM25 + graph) |
|---|---:|---:|
| Claims (total) | 157 | 158 |
| Claims per answer | 3.57 | 3.59 |
| Structural citation precision | 1.000 | 0.996 |
| **Claims supported by cited evidence** | **0.389** | **0.418** |
| Claims contradicted by cited evidence | 0.038 | 0.070 |

Per-query support difference (system − vector): +0.046, 95% CI [−0.068, 0.157], Monte Carlo p = 0.43, so no significant difference. Support is higher than with the 3B generator (§5c, about 0.30), but those runs used different queries (14 GNN test queries), so the change is indicative only. Structurally valid citations still do not imply support: about 60% of claims are not entailed by their cited passages under the automated NLI check, which awaits human validation.

## 6. Data and graph facts

- 3.6% of 625,222 references resolve inside the corpus; 71.4% of papers have no citation edge.
- **Densification (2026-11-15, `src/ingestion/citation_densify.py`).** Only 10,899 papers (23%) had reference lists; Semantic Scholar hides most lists for publisher-licensing reasons. Added reference lists came from the Semantic Scholar batch API (21,027 papers) and OpenAlex by DOI (44,090 of 46,198 matched). Opt-in layers, each its own label or edge type, so defaults and frozen results are unchanged:

| | Before | After |
|---|---:|---:|
| Papers with a reference list | 10,899 | 46,708 |
| Corpus-internal citation edges | 22,370 `CITES` | + 133,527 `CITES_ADDED` |
| Bibliographic coupling (`COUPLED_WITH`, cosine, top 10 per paper, ≥ 2 shared) | 0 | 411,418 |
| External papers cited by ≥ 10 corpus papers (`ExternalPaper`, `CITES_EXTERNAL`) | 0 | 24,714 nodes, 641,616 edges |
| Papers with no citation link | 33,995 (71.4%) | 6,394 (13.4%); 1,154 (2.4%) counting coupling |
- 51 near-duplicate paper pairs (cosine ≥ 0.995), mostly preprint/published versions.
- Before flags, 83.5% of spaCy concepts were singletons and "this paper" linked 7,013 papers; 32,789 generic concepts are now flagged.
- 238,095 `SIMILAR_TO` edges (top-5 SPECTER neighbours; median cosine 0.907).

## 7. Open items for publication

Done: human relevance sample and judge validation (§2b); stratified 44-query benchmark (§2d); citation densification (§6); matched-budget vector RAG vs this system (§5c).

1. **Second human rater** for relevance (human–human κ), round-1 hypotheses, and round-1 contradictions.
2. **Round 2 human ratings** (`evaluation/human_review/*round2*`): revised hypothesis method and citation-grounded contradiction index on held-out queries.
3. Retrieval on the densified graph (`CITES_ADDED`, `COUPLED_WITH`) as a new variant: dev selection on the 22 stratified dev queries, pool extension, then test once.
4. Optional robustness check: regrade both pools with a larger judge (27B class) and confirm the §2c/§2d conclusions hold.
5. Deduplication of 51 near-duplicate pairs; contradiction benchmark with ≥ 10 in-domain positives.
