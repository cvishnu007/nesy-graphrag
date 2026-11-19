# Where Does Graph Structure Help Scholarly RAG? Topical Retrieval, Citation Relatedness, and Research Discovery

*Draft, 2026-11-15. Every number cites a section of [RESULTS.md](RESULTS.md); ⏳ marks results pending human ratings. Authors, affiliations, and venue formatting to be added.*

## Abstract

Graph-based retrieval-augmented generation (GraphRAG) is increasingly proposed for scientific literature, but it is rarely tested against strong text baselines with controlled protocols. We build a literature assistant over 47,619 computer-science abstracts that combines dense retrieval, BM25, a citation graph densified from Semantic Scholar and OpenAlex (corpus-internal citation edges ×7), and a SciERC entity graph. We evaluate it on two public benchmarks and two project benchmarks graded by an LLM judge validated against human relevance grades. Three findings emerge. (1) For topical queries, a general embedding with BM25 fusion beats every scientific embedding model tested and more than doubles the original system's NDCG@10 (0.644 vs 0.287 on 44 topic-stratified queries). Graph expansion adds no significant gain, even after densification, because the papers it introduces are exactly as relevant as those they displace. (2) For citation recommendation, co-citation over the densified graph improves Recall@20 by 28% relative to text retrieval (p < 0.0001). (3) In a time-sliced discovery benchmark, graph signals nearly double average precision over text similarity when predicting which method–task combinations researchers adopt after a cutoff. We also report negative results: a small LLM relevance judge with position bias, structural-hole hypotheses rated below a nearest-neighbour baseline, and NLI contradiction screening on abstracts with no true positives among 56 top candidates. Together these locate where graph structure helps scholarly RAG: citation-level relatedness and discovery, not topical ranking.

## 1. Introduction

Literature review assistants must find relevant work, surface disagreements, and suggest new directions. GraphRAG systems add a knowledge graph to retrieval on the premise that structure supplies what text similarity misses (Edge et al., 2024). For scholarly corpora the natural graphs are citations and extracted scientific entities. Whether they help is usually asserted rather than measured, and published comparisons rarely use strong text baselines, frozen development/test protocols, or validated judges.

We take an empirical stance and ask: *for which tasks does graph structure add value over a strong text retriever?* Our contributions:

1. **A controlled evaluation** of retrieval for scholarly RAG on SciDocs, LitSearch, and two project benchmarks, using a dev/test protocol, pool extension, and an LLM judge validated on human grades (§4).
2. **A boundary result for graph structure.** It is redundant for topical ranking even after densifying citations 7× (§5.2), but it gives large gains for citation recommendation (§5.3) and for predicting future method–task combinations (§5.4).
3. **A methodological warning about LLM judges.** A 3B judge's labels agreed with a human at κ 0.17 because of batch-position bias. Correcting the judge changed every effect size (§4.3).
4. **Negative results with diagnoses** for hypothesis generation and contradiction detection on abstracts, plus revised methods (§5.5–5.6).

## 2. Related work

**GraphRAG.** Edge et al. (2024) build LLM-extracted entity graphs with community summaries for query-focused summarisation. We use citation and supervised entity graphs, and we measure task by task where they help.

**Scientific retrieval and RAG.** SPECTER (Cohan et al., 2020) and SciNCL (Ostendorff et al., 2022) learn citation-informed embeddings. OpenScholar (Asai et al., 2024) retrieves over about 45M papers. LitSearch (Ajith et al., 2024) and SciDocs (BEIR; Thakur et al., 2021) benchmark literature search. We find a general embedding (BGE; Xiao et al., 2023) with BM25 beats the scientific embeddings tested.

**Contradictions and claims.** PaperQA2 (Skarlinski et al., 2024) reports contradiction detection validated by experts in biology. SciFact (Wadden et al., 2020) provides expert-labelled claim–evidence pairs, which we use to validate verdict models.

**Literature-based discovery (LBD).** LBD predicts implicit connections from co-occurrence (ABC models) and evaluates with time slicing: predictions from data before a cutoff are checked against later literature. We apply this protocol to method–task pairs extracted with a SciERC-trained tagger (Luan et al., 2018).

**Attribution.** ALCE (Gao et al., 2023) evaluates citation precision and NLI support, which we adopt for answer grounding.

*(References still to verify against original records: SPECTER, SPECTER2, BEIR, SciFact, SciERC, BGE, GLiNER, Sakai 2007; see PAPER_OUTLINE §8.)*

## 3. System

**Corpus.** 47,619 Semantic Scholar abstracts (11 CS topics, 2020–2025).

**Retrieval.** BGE-base-en-v1.5 dense retrieval with a query instruction, BM25, and graph expansion (`CITES`, `SIMILAR_TO`; relative gate: candidates must be within 0.03 of the weakest seed's similarity), fused by weighted reciprocal-rank fusion (RRF).

**Citation densification (RESULTS §6).** Only 23% of papers came with reference lists. Fetching lists from the Semantic Scholar batch API and OpenAlex (DOI-matched; 44,090 of 46,198) raised coverage to 98%. Added layers: 133,527 corpus-internal citations, 411,418 bibliographic-coupling edges, and 24,714 external papers. Isolated papers fell from 71.4% to 2.4%.

**Entities (RESULTS §3).** SciBERT fine-tuned on SciERC (test typed F1 0.639) extracts Method, Task, Material, Metric, and OtherScientificTerm mentions. The graph has 323,624 `SciEntity` nodes and 696,390 mention edges.

**Reasoning modes.**
- **Grounded review:** sentence-level passage IDs; invalid citations withheld.
- **Contradiction:** NLI screening plus a structured LLM verdict, or a corpus-wide citation-grounded index.
- **Hypothesis:** structural holes, a problem bridge, or a method–task ranker.
- **Related work:** co-citation fusion.

## 4. Evaluation design

### 4.1 Benchmarks

| Benchmark | Queries | Labels |
|---|---|---|
| SciDocs (BEIR) | 1,000 | citation and co-view derived |
| LitSearch | 597 | author-inspected targets |
| GNN project benchmark | 20 (6 dev, 14 test) | LLM judge, regraded (§4.3) |
| **Topic-stratified benchmark** | 44 (4 per topic; 22 dev, 22 test) | LLM judge; queries team-reviewed and frozen before pooling |
| Citation recommendation | 600 papers (200 dev, 400 test) | real reference lists |
| Time-sliced discovery | 120 tasks, ~4,600 candidates each | later co-occurrence |

### 4.2 Protocol

- **Dev first, test once.** Configurations are selected on dev and tested once; selection rules are written down before results are seen.
- **Pool extension.** Unjudged results of new variants are graded by the same judge, so every scored list is fully judged.
- **Statistics.** Paired randomisation tests (exact for ≤ 20 pairs, Monte Carlo otherwise) with bootstrap 95% CIs, and Holm correction where several comparisons are predeclared.
- **Metrics.** Implementations match `trec_eval` exactly (unit-tested).

### 4.3 Validating the relevance judge (RESULTS §2b)

One annotator graded 200 pooled pairs, blinded to method and AI grade. Judges were selected on half the pairs and scored once on the other half. A second, independent annotator graded a blinded 50-pair stratified subset of the held-out half: human–human κ 0.91, against 0.76–0.80 for the 7B judge and 0.86–0.95 for the 27B judge on the same pairs.

| Judge | Held-out quadratic-weighted κ |
|---|---:|
| Frozen pool labels (qwen2.5-3B, 4 papers per call) | 0.17 (all 200) |
| qwen2.5-3B re-run | 0.44 |
| **qwen2.5-7B, strict rubric, 1 paper per call (used)** | **0.75** |
| Qwen3.8-27B, strict rubric | 0.88 |

The 3B judge's labels depended on batch position. It used 20 of 81 possible four-grade patterns and gave the second paper a 2 in almost every batch. Grading one paper per call raised the 7B judge from κ 0.60 to 0.77 on the selection half. Every project retrieval number below uses the validated 7B labels.

## 5. Results

### 5.1 Retrieval models (RESULTS §1)

BGE beats SPECTER, SPECTER2 (base, without adapter), SciNCL, and OpenScholar on both public benchmarks (all p < 0.001 on LitSearch). BM25 fusion helps every dense model on LitSearch (BGE + BM25: NDCG@10 0.490).

### 5.2 Topical retrieval: graph structure is redundant (RESULTS §2c–2e)

| 44 stratified queries | NDCG@10 |
|---|---:|
| BGE + BM25 + graph (selected Nov 13) | **0.644** |
| BM25 | 0.626 |
| BGE + BM25 | 0.621 |
| BGE | 0.593 |
| Original system (SPECTER) | 0.287 |

Selected vs original: +0.358 [0.285, 0.434]. Selected vs BGE alone: +0.052 (p = 0.010; significant after Holm correction). Graph over BGE + BM25: +0.023, not significant.

With the densified citation layers (selected on dev, tested once), the graph adds −0.003 on test (p = 0.92). Papers introduced by dense expansion have the same mean relevance as those displaced (1.19 vs 1.21 on 0–2; 51% grade 2 in both). Text retrieval already fills the top 10 with relevant papers (P@10 ≈ 0.78), so equally relevant substitutes cannot raise the score.

### 5.3 Citation recommendation: graph structure helps (RESULTS §2f)

Given a paper's title and abstract, rank the corpus papers it cites. Leakage controls:
- the query paper's own edges are never used;
- papers citing it are excluded from the neighbour set;
- every method ranks only papers published no later than it.

| 400 test papers | Recall@20 | NDCG@10 |
|---|---:|---:|
| **BGE + BM25 + co-citation graph** | **0.308** | **0.283** |
| BGE | 0.251 | 0.231 |
| BGE + BM25 | 0.241 | 0.217 |

+0.068 Recall@20 over BGE + BM25 [0.055, 0.081], p < 0.0001.

### 5.4 Time-sliced discovery: graph signals predict future research (RESULTS §4c–4d)

From papers up to 2022 we rank, for each task, the methods never yet combined with it, and check against 2023–2025 papers (≥ 2 confirming papers).

| Scorer (120 tasks) | AP | P@10 | Recall@100 |
|---|---:|---:|---:|
| Vector + entity bridge + citation bridge (RRF) | **0.083** | **0.067** | **0.406** |
| Entity bridge (Adamic–Adar) | 0.072 | 0.061 | 0.398 |
| Vector similarity | 0.044 | 0.037 | 0.302 |
| Popularity | 0.032 | 0.023 | 0.288 |
| Random | 0.003 | 0.003 | 0.024 |

Vector + graph vs vector: +0.039 AP [0.022, 0.056], p < 0.0001. The result holds with a 2021 cutoff (+0.045) and with ≥ 3 confirming papers (+0.036, p = 0.035).

Used as generators, ranking method–task pairs directly (P@10 0.067) beats the methods of vector-neighbour papers (0.038). A paper-candidate "bridge" generator that requires methods unseen among the 15 nearest papers scores below that baseline (0.021). The filter discards the methods most often adopted next.

### 5.5 Hypothesis candidates rated by humans (RESULTS §4b)

Round 1 (30 queries, two raters, blinded; inter-rater κ 0.82–0.90): the original structural-hole candidates were rated less relevant than vector neighbours (3.10 vs 4.12 of 5, p = 0.001), less promising (2.15 vs 2.67, p = 0.03), and no more novel (1.88 vs 2.17, n.s.). The rated variant ran without the generic-entity filter. Round 2 (14 held-out queries, four blinded arms, filter on, one rater): the revised problem-bridge generator, preferred by the LLM judge on the selection queries, was no better than the original and did not beat vector neighbours (relevance 3.54 vs 3.93, promise 3.31 vs 3.64; n.s. with 13 pairs). The original again scored lower on promise (3.00 vs 3.64, p = 0.03). Paper-level graph candidates thus lose to vector similarity in both rounds; only the method–task ranker (§5.4) shows a graph advantage, on the time-sliced benchmark.

### 5.6 Contradictions (RESULTS §5b, §5b-2)

**Verdict models on SciFact (338 pairs).**

| Model | F1 |
|---|---:|
| qwen2.5-3B | 0.53 |
| qwen2.5-7B | 0.70 |
| Qwen3.8-27B | 0.79 |
| GPT-OSS-120B | 0.81 |

**In-corpus, round 1.** Two raters (κ 0.76) labelled 68 NLI-screened pairs. No pair is a contradiction for both; one rater called 2 of the 56 top pairs contradictions (both LLM versus search-based unit-test generation) and the other called them tensions; 7 of 56 (12.5%) are a tension or contradiction for at least one rater, against 0 of 12 controls. The screener mostly pairs papers that each report their own gains. A 7B verdict filter cuts the pairs shown from 68 to 5: it keeps both disputed pairs, and the other 3 (about 5% of negatives) are false alarms.

**Revision.** A citation-grounded index pairs critical findings ("fails to", "contrary to", "surprisingly") with the papers they cite: 591 candidates, 9 of the top 150 flagged by the verdict model. ⏳ Round 2 precision (40 pairs).

### 5.7 Answer grounding (RESULTS §5c)

With a 3B generator, about 30% of claims were supported by their cited passages, despite about 99% structurally valid citations. Vector RAG and our system did not differ (p = 0.69). With the 7B generator on all 44 stratified queries, support rises to 38.9% (vector RAG) and 41.8% (our system), with no significant difference (+0.046 [−0.068, 0.157], p = 0.43; RESULTS §5c-2). About 60% of structurally valid claims are still not entailed by their cited passages, which argues for semantic support checks rather than passage-ID validation alone.

## 6. Discussion

**Topical retrieval for short queries saturates with text.** The graph substitutes relevant papers rather than adding new ones.

**Citation structure carries different information.** When the target is citation-level relatedness or future method adoption, which is what "who builds on whom" encodes, graph signals add large, significant gains, and they complement text.

**The way a signal is used matters.** The same entity graph that predicts future combinations (§5.4) produced poor hypotheses when routed through candidate papers with a novelty filter (§5.4, §5.5).

**Judges must be validated.** An unvalidated small judge would have produced wrong effect sizes throughout.

## 7. Limitations

- Abstracts only; one corpus (CS, 2020–2025, 47,619 papers).
- Project relevance labels come from an LLM judge (held-out κ 0.75 against the first annotator). Human–human agreement was measured on only 50 pairs (κ 0.91), with two annotators.
- "Confirmed" in the time-sliced benchmark means later co-occurrence in abstracts, not a successful result. Entity extraction is automatic (F1 0.64).
- SciFact is biomedical; in-corpus contradiction precision rests on small human samples.
- SPECTER2 was evaluated without its query adapter.
- Hypotheses suggested as "never combined in the corpus" may be known in older literature.

## 8. Conclusion

Graph structure is not a general upgrade for scholarly RAG. It is redundant for topical ranking but adds substantial value for citation recommendation and research discovery. Evaluations of GraphRAG should name the task, use strong text baselines, and validate their judges.

## Reproducibility

Code, configurations, frozen query sets, judge prompts, and result artifacts are in the repository (`src/evaluation/*`, `results/*`). The commands are listed in [AGENT_HANDOFF.md](../AGENT_HANDOFF.md). Data licences: Semantic Scholar abstracts (ODC-By), OpenAlex (CC0), SciFact (CC BY 4.0); LitSearch and SciDocs under their own terms.
