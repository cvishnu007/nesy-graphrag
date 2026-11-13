# Results (November 13–14, 2026)

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

## 2. Project benchmark (20 GNN queries; AI-reference labels; pool-extended)

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

## 5. Contradiction candidates

| Generator | Queries with candidates (of 26) | Evidence |
|---|---:|---|
| Concept overlap, spaCy raw | 23 | none; pairs driven by "this paper", "graph neural networks" |
| Concept overlap, noise removed | 0 (Nov 13 probe) to 10 | none |
| **NLI sentence pairs, author findings only** | **18** | two sentences with passage IDs |

Structured JSON verdicts: 5/5 valid in the end-to-end smoke test (legacy line format: 17 of 39 malformed on the frozen test set). The local 3B judge reports confidence 1.0 for every verdict, so confidence is not calibrated; contradiction accuracy is unmeasured until a benchmark with ≥ 10 positives per split exists.

## 6. Data and graph facts

- 3.6% of 625,222 references resolve inside the corpus; 71.4% of papers have no citation edge.
- 51 near-duplicate paper pairs (cosine ≥ 0.995), mostly preprint/published versions.
- Before flags, 83.5% of spaCy concepts were singletons and "this paper" linked 7,013 papers; 32,789 generic concepts are now flagged.
- 238,095 `SIMILAR_TO` edges (top-5 SPECTER neighbours; median cosine 0.907).

## 7. Open items for publication

1. Human labels: 200-pair relevance sample (judge agreement), hypothesis ratings with weighted kappa.
2. Contradiction benchmark with ≥ 10 positives per split; calibrate or replace LLM verdict confidence.
3. Topic-stratified 44-query set (draft in `evaluation/benchmarks/`) reviewed, pooled, and graded.
4. Corpus densification by reference closure; deduplication.
5. Matched-budget vector RAG vs GraphRAG answer evaluation.
