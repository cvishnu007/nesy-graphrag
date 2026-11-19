# Where Does Graph Structure Help Scholarly RAG? Topical Retrieval, Citation Relatedness, and Research Discovery

*Draft, 2026-11-20. Every number cites a section of [RESULTS.md](RESULTS.md). Pending: figures, checking the references not marked †, authors, affiliations, and venue formatting.*

## Abstract

Graph-based retrieval-augmented generation (GraphRAG) is increasingly proposed for scientific literature, but it is rarely tested against strong text baselines with controlled protocols. We build a neuro-symbolic literature assistant over 47,619 computer-science abstracts: neural components (dense retrieval, a local LLM, NLI, a fine-tuned entity tagger) propose, and symbolic components (a typed citation and SciERC entity graph, and rule-based provenance and citation constraints) restrict and verify what is shown. We evaluate it on four public benchmarks (SciDocs, LitSearch, DBLP, PubMed) and on project benchmarks graded by an LLM judge validated against human relevance grades (κ 0.75 on one topic; 0.57 on ten others, where it is systematically strict, while a larger judge holds κ 0.88). Four findings emerge. (1) For topical queries, a general embedding (BGE) has the best NDCG@10 of the encoders tested on both public retrieval benchmarks, and on 44 topic-stratified queries BM25 (0.626) and BGE with BM25 (0.621) are already strong. Graph expansion adds no significant gain (0.644), even after densifying citations, because the papers it introduces are about as relevant as those they displace, and a LightRAG-style entity-graph retriever scores far lower (0.396). (2) For citation recommendation, expanding text retrieval with the papers cited by the query's nearest neighbours, an idea going back to Strohman et al. (2007) and used for candidate selection by Citeomatic, improves Recall@20 by 28% relative to text retrieval (p < 0.0001). On the public DBLP and PubMed benchmarks, with no training on either dataset, it raises Recall@20 on top of every encoder tested, including the citation-trained SPECTER2 and SciNCL. With SPECTER2 on DBLP the result is close to the trained Citeomatic system (F1@20 0.300 vs 0.303; MRR 0.708 vs 0.689), although SPECTER2 was itself pretrained on citation links from a corpus that may overlap with DBLP. In biomedicine it lowers the rank of the first correct paper. (3) In a time-sliced discovery benchmark, adding graph signals to text similarity nearly doubles average precision when predicting which method–task combinations researchers adopt after a cutoff. (4) For contradiction detection, restricting candidates to citation links between critical findings and the papers they dispute raises contradiction-or-tension precision from 9% (5 of 56) for NLI screening to 56% (5 of 9; 95% CI 27–81%) after an LLM verdict filter, in a small single-rater study. We also report negative results: a small LLM relevance judge with position bias, and graph-derived hypothesis candidates rated below a nearest-neighbour baseline by humans in two rounds, including on held-out queries. Together these locate where graph structure helps scholarly RAG: citation-level relatedness and discovery, not topical ranking.

## 1. Introduction

Literature review assistants must find relevant work, surface disagreements, and suggest new directions. GraphRAG systems add a knowledge graph to retrieval on the premise that structure supplies what text similarity misses (Edge et al., 2024). For scholarly corpora the natural graphs are citations and extracted scientific entities. Whether they help is usually asserted rather than measured, and published comparisons rarely use strong text baselines, frozen development/test protocols, or validated judges.

We take an empirical stance and ask: *for which tasks does graph structure add value over a strong text retriever?* Our contributions:

1. **A controlled, task-by-task evaluation** of graph structure in scholarly RAG on four public benchmarks (SciDocs, LitSearch, DBLP, PubMed) and four project benchmarks, using a dev/test protocol, pool extension, paired tests, and an LLM judge validated on human grades (§4).
2. **A boundary result for graph structure.** It is redundant for topical ranking even after densifying citations 7×, and a LightRAG-style entity-graph retriever falls far below text retrieval (§5.2). For citation recommendation, the neighbour-citation expansion we use is not new (Strohman et al., 2007; Bhagavatula et al., 2018); our contribution is showing that it still adds to modern encoders, including ones trained on citations (§5.3). Graph signals also improve prediction of future method–task combinations (§5.4).
3. **A methodological warning about LLM judges.** A 3B judge's labels agreed with a human at κ 0.17 because of batch-position bias. Correcting the judge changed the main effect sizes substantially (§4.3).
4. **A neuro-symbolic assistant with human-rated reasoning modes** (§3.1). Citation-grounded contradiction detection works where NLI screening on abstracts did not (§5.6). Graph-derived hypothesis candidates do not beat vector neighbours in two human rounds, and a revision chosen with an LLM judge did not transfer to held-out queries (§5.5).

## 2. Related work

**GraphRAG.** Edge et al. (2024) build LLM-extracted entity graphs with community summaries for query-focused summarisation. LightRAG (Guo et al., 2024) indexes LLM-extracted entities and relations and retrieves through dual-level keyword matching; we evaluate its retrieval algorithm as a baseline. HippoRAG (Gutiérrez et al., 2024) runs personalised PageRank over an LLM-built graph for multi-hop QA. HippoRAG 2 (Gutiérrez et al., 2025) adds deeper passage integration. Peng et al. (2024) survey the area. Two recent studies ask our question for general corpora: Han et al. (2025) find RAG better on single-hop and detail questions and GraphRAG better on multi-hop reasoning, and GraphRAG-Bench (Xiang et al., 2026) finds basic RAG matching or beating GraphRAG on simple fact retrieval. These systems build their graphs with LLMs and target open-domain QA; we use citation and supervised entity graphs over scholarly text and measure, task by task, where they help.

**Neuro-symbolic AI.** Neuro-symbolic systems combine learned models with explicit symbols, rules or structured knowledge (Garcez and Lamb, 2023; Kautz, 2022). Our system is a loose, pipelined coupling (the "Neuro | Symbolic" pattern in Kautz's taxonomy): neural models propose candidates, and a typed graph and hand-written rules constrain and verify them (§3.1). We do not learn rules or train end to end.

**Scientific retrieval and RAG.** SPECTER (Cohan et al., 2020), SciNCL (Ostendorff et al., 2022) and SPECTER2 (Singh et al., 2023) learn citation-informed paper embeddings. OpenScholar (Asai et al., 2024) retrieves over about 45M papers. LitSearch (Ajith et al., 2024) and SciDocs (Cohan et al., 2020; in BEIR, Thakur et al., 2021) benchmark literature search. We find that a general embedding (BGE; Xiao et al., 2024) with BM25 (Robertson and Zaragoza, 2009), fused by reciprocal rank (Cormack et al., 2009), beats the scientific embeddings tested.

**Citation recommendation.** Co-citation (Small, 1973) and bibliographic coupling (Kessler, 1963) are the classical citation-based relatedness measures. McNee et al. (2002) recommend citations by collaborative filtering over the citation graph. Strohman et al. (2007) retrieve the papers most similar to a manuscript and add every paper they cite, which is the same expansion as our co-citation step; Citeomatic's NNSelect (Bhagavatula et al., 2018) likewise takes candidates from the query's nearest neighbours in embedding space and their citations, and then trains a reranker. ClusCite (Ren et al., 2014) and Citeomatic define the DBLP and PubMed benchmarks we reuse with their published protocol. Our citation-recommendation method is therefore not new. What we add is (i) evidence that this expansion still helps on top of modern encoders, including citation-trained ones, with no training on the target data, and (ii) the same signal used inside a RAG assistant, where it explains each suggestion.

**Contradictions and citation function.** Citation-function analysis classifies why a paper cites another, including to criticise or contrast (Teufel et al., 2006; Jurgens et al., 2018; Cohan et al., 2019). scite (Nicholson et al., 2021) applies such a classifier to citation statements from full text at scale, labelling them supporting, contrasting or mentioning. Our citation-grounded contradiction mode is a light, abstract-only version of the same idea: the citing sentence's critical wording is a filter rather than a trained classifier, and an LLM verdict decides. PaperQA2 (Skarlinski et al., 2024) reports contradiction detection validated by experts in biology. SciFact (Wadden et al., 2020) provides expert-labelled claim–evidence pairs, which we use to validate verdict models.

**Literature-based discovery (LBD).** LBD predicts implicit connections from co-occurrence (Swanson, 1986) and evaluates with time slicing: predictions from data before a cutoff are checked against later literature. AGATHA (Sybrandt et al., 2020) does this for biomedicine with graph mining and transformers, and Science4Cast (Krenn et al., 2023) frames future AI research directions as link prediction in a concept network, finding hand-crafted network features strongest. SciMON (Wang et al., 2024) generates natural-language ideas optimised for novelty, and Si et al. (2025) find in a large blind human study that LLM-generated ideas are judged more novel but slightly less feasible than experts'. We apply the time-slicing protocol to method–task pairs extracted with a SciERC-trained tagger (Luan et al., 2018; SciBERT, Beltagy et al., 2019), scored with Adamic–Adar (Adamic and Adar, 2003) and citation bridges.

**Attribution and LLM judges.** ALCE (Gao et al., 2023) evaluates citation precision and NLI support, which we adopt for answer grounding. LLMs can match human relevance labellers when the prompt is chosen against human labels (Thomas et al., 2024; reproduced openly as UMBRELA, Upadhyay et al., 2024), and LLM judges show position bias (Zheng et al., 2023), which we also observe. Agreement uses weighted kappa (Cohen, 1968); multiple comparisons use Holm's procedure (Holm, 1979).

## 3. System

**Corpus.** 47,619 Semantic Scholar abstracts (Kinney et al., 2023; 11 CS topics, 2020–2025).

**Language model.** `qwen2.5:7b-instruct` (Qwen Team, 2024), run locally.

**Retrieval.** BGE-base-en-v1.5 dense retrieval with a query instruction, BM25, and graph expansion (`CITES`, `SIMILAR_TO`; relative gate: candidates must be within 0.03 of the weakest seed's similarity), fused by weighted reciprocal-rank fusion (RRF).

**Citation densification (RESULTS §6).** Only 23% of papers came with reference lists. Fetching lists from the Semantic Scholar batch API and OpenAlex (Priem et al., 2022; DOI-matched; 44,090 of 46,198) raised coverage to 98%. Added layers: 133,527 corpus-internal citations, 411,418 bibliographic-coupling edges, and 24,714 external papers. Isolated papers fell from 71.4% to 2.4%.

**Entities (RESULTS §3).** SciBERT fine-tuned on SciERC (test typed F1 0.639) extracts Method, Task, Material, Metric, and OtherScientificTerm mentions. The graph has 323,624 `SciEntity` nodes and 696,390 mention edges; it keeps the first 15 mentions per abstract. Over all mentions there are 558,123 distinct names, which the LightRAG-style baseline (§5.2) uses.

**Reasoning modes.**
- **Grounded review:** sentence-level passage IDs; invalid citations withheld.
- **Contradiction:** NLI screening (a DeBERTa-v3 cross-encoder; He et al., 2023) plus a structured LLM verdict, or a corpus-wide citation-grounded index.
- **Hypothesis:** structural holes, a problem bridge, or a method–task ranker.
- **Related work:** co-citation fusion.

### 3.1 Neural and symbolic components

The "neuro-symbolic" in the project name refers to a division of labour, not to learned logic. Neural models propose; explicit structure and rules decide what is allowed and verify it.

| Step | Neural (learned) | Symbolic (explicit structure or rule) |
|---|---|---|
| Knowledge | BGE embeddings; SciBERT–SciERC tagger extracts entities | Typed graph in Neo4j: `Paper`, `SciEntity` (Method, Task, Material, Metric, Other), `CITES`, `CITES_ADDED`, `COUPLED_WITH`, `MENTIONS` |
| Retrieval | Dense and BM25 ranking | Graph expansion only along typed edges, with a similarity gate; fixed RRF fusion |
| Review answers | LLM writes claims | Every claim must cite sentence-level passage IDs that exist in the retrieved set; claims with missing or unknown IDs are withheld and listed, and cited paper IDs must exist in the graph |
| Contradictions | NLI screening; LLM verdict | Candidates must be joined by a citation edge, the citing sentence must match a critical-language pattern, and both sentences must name the same specific entity |
| Hypotheses | Vector similarity between entity contexts | Only Method–Task pairs that never co-occur; Adamic–Adar over shared entities and citation links between the two sets of papers; generic-entity exclusions |
| Related work | Nearest neighbours | Suggestions come from the neighbours' citation records, shown as an explanation for each suggestion |

This coupling is what makes outputs checkable: each answer, contradiction or hypothesis is traceable to graph facts and passage IDs. It does not make them correct; §5.7 shows that structurally valid citations are often not semantically supported.

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

The 3B judge's labels depended on batch position. It used 20 of 81 possible four-grade patterns and gave the second paper a 2 in almost every batch. Grading one paper per call raised the 7B judge from κ 0.60 to 0.77 on the selection half. Every project retrieval number below uses the validated 7B labels. The 200 validation pairs come from the GNN benchmark pool. A third rater (a project member who did not build the system; no AI assistance, blind to the judge's grades) graded 100 further pairs from the other ten topics of the 44-query benchmark: there the judge reaches κ 0.57 (exact 61%, binary 74%), below our 0.6 bar. Its errors are one-sided (stricter than the human on 32 items, more lenient on 7) and equally large for the systems with and without graph expansion (+0.40 grade on sampled top-10 papers), so graph comparisons are unaffected; but it is stricter still on the original system's papers (+0.61), so the gain over the original system is likely overstated, and absolute NDCG values are underestimates (RESULTS §2b). A larger judge transfers: Qwen3.8-27B (Groq, one paper per call, run 2026-11-20) reaches κ 0.88 on the same 100 pairs (exact 84%, binary 95%), as on the GNN pairs, and its errors are small and two-sided (5 higher, 11 lower than the human). Regrading the 44-query pool with it (2,305 pairs) is the remaining fix; on the free API tier it takes about 12 days.

## 5. Results

### 5.1 Retrieval models (RESULTS §1)

BGE has the best NDCG@10 among SPECTER, SPECTER2 (base, without adapter), SciNCL, and OpenScholar on both public benchmarks (all p < 0.001 on LitSearch), though SciNCL has slightly higher Recall@100 on SciDocs (0.508 vs 0.493). BM25 fusion helps every dense model on LitSearch (BGE + BM25: NDCG@10 0.490) but lowers BGE on SciDocs (0.218 to 0.198).

### 5.2 Topical retrieval: graph structure is redundant (RESULTS §2c–2e)

| 44 stratified queries | NDCG@10 |
|---|---:|
| BGE + BM25 + graph (selected Nov 13) | **0.644** |
| BM25 | 0.626 |
| BGE + BM25 | 0.621 |
| BGE | 0.593 |
| Original system (SPECTER) | 0.287 |

Selected vs original: +0.358 [0.285, 0.434]; the judge under-grades the original system's papers more than ours (§4.3), so this gap is likely overstated. Selected vs BGE alone: +0.052 [0.012, 0.087] (p = 0.0104 with 1,000,000 sign-flip permutations (two seeds agree; the earlier 20,000-permutation estimate was 0.0095); Holm-adjusted p ≈ 0.052 over the nine predeclared comparisons, so not significant). Graph over BGE + BM25: +0.023, not significant.

With the densified citation layers (selected on dev, tested once), the graph adds −0.003 on test (p = 0.92). Papers introduced by dense expansion have the same mean relevance as those displaced (1.19 vs 1.21 on 0–2; 51% grade 2 in both). Text retrieval already fills the top 10 with relevant papers (P@10 ≈ 0.78), so equally relevant substitutes cannot raise the score.

**GraphRAG baseline (RESULTS §2g).** A LightRAG-style retriever (LightRAG's query algorithm over our SciERC entity graph; the original LLM-built index is infeasible at this scale on our hardware) reaches 0.396 NDCG@10 in its default mix mode and 0.176 in hybrid mode, against 0.642 for this system (0.644 in the table above; the extended pool adds relevant papers to the ideal ranking) (+0.246, 39 wins and 3 losses, p < 0.0001) and 0.591 for BGE alone. Entity-centric retrieval ranks papers by which entities they mention; generic keywords such as "scalability" match entities found in thousands of papers. We report this as a re-implementation over a supervised graph, not the original package.

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

**Public benchmarks (RESULTS §2h).** On the DBLP and PubMed benchmarks of Bhagavatula et al. (2018), the same method, with no training on either dataset, gives the results below. The expansion is close to Citeomatic's NNSelect candidate step (§2), so this tests whether an old idea still adds to modern encoders, not a new method.

| Test split | DBLP F1@20 / MRR | PubMed F1@20 / MRR |
|---|---|---|
| BGE + BM25 | 0.230 / 0.649 | 0.290 / 0.747 |
| **+ co-citation** | **0.284 / 0.677** | **0.305** / 0.699 |
| ClusCite (published) | 0.237 / 0.548 | 0.274 / 0.578 |
| Citeomatic, best (published) | 0.303 / 0.689 | 0.329 / 0.771 |

On DBLP, co-citation adds +0.069 Recall@20 and +0.028 MRR (both significant) and comes close to the trained Citeomatic system. The gain is not specific to BGE. With citation-trained encoders (test, F1@20 / MRR, + BM25 → + BM25 + co-citation), SPECTER2 goes from 0.236 / 0.663 to **0.300 / 0.708**, and SciNCL from 0.230 / 0.629 to 0.283 / 0.666. Co-citation adds +0.069 to +0.081 Recall@20 for all three encoders (p < 0.001). So, on DBLP, the citation graph carries information that these citation-trained embeddings do not already capture. Two caveats: SPECTER2 and SciNCL were pretrained on citation links from Semantic Scholar, which probably includes many DBLP and PubMed papers, so they are not training-free; and the comparison is with 2018 systems. On PubMed it adds +0.010 to +0.026 Recall@20 across the three encoders, but lowers MRR by 0.023 (SPECTER2) to 0.056 (SciNCL). So the citation-graph benefit is clear in CS and mixed in biomedicine. A diagnosis on the validation year shows why. Co-citation favours widely cited papers in both datasets, but on PubMed text retrieval already ranks a correct paper first 71% of the time against 45% for the graph alone, so equal-weight fusion displaces correct first picks; on DBLP the two are equally precise at the top (52%).

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

**Revision.** A citation-grounded index pairs critical findings ("fails to", "contrary to", "surprisingly") with the papers they cite: 591 candidates, 9 of the top 150 flagged by the verdict model. **Round 2** (40 blinded pairs, one rater, the system's developer): 5 of the 9 verdict-flagged pairs (56%; 95% Wilson CI 27–81%) are a contradiction or tension, including 2 real contradictions (a follow-up reporting that Post2Vec does not improve over the state of the art; constructive exact low-rank embeddings against an impossibility result), against 9% (5 of 56, same rater) and 0 for the round 1 screener (Fisher p = 0.003 and 0.017). Round 1 and round 2 use different queries and pair sources, so this cross-round comparison is indicative, and the 56% rests on 9 pairs. NLI rank within the index matters (top 150: 10 of 24 vs rank 151+: 1 of 16, p = 0.03); the verdict filter's extra gain (56% vs 33%) is not significant at this size.

### 5.7 Answer grounding (RESULTS §5c)

With a 3B generator, about 30% of claims were supported by their cited passages, despite about 99% structurally valid citations. Vector RAG and our system did not differ (p = 0.69). With the 7B generator on all 44 stratified queries, support rises to 38.9% (vector RAG) and 41.8% (our system), with no significant difference (+0.046 [−0.068, 0.157], p = 0.43; RESULTS §5c-2). About 60% of structurally valid claims are still not entailed by their cited passages, which argues for semantic support checks rather than passage-ID validation alone.

## 6. Discussion

**Topical retrieval for short queries saturates with text.** This agrees with GraphRAG-Bench and Han et al. (2025) on general corpora. The graph substitutes relevant papers rather than adding new ones, and retrieving through entities instead of text loses relevance, because a paper's entities say little about whether it addresses the query.

**Citation structure carries different information.** When the target is citation-level relatedness or future method adoption, which is what "who builds on whom" encodes, graph signals add large, significant gains, and they complement text.

**The way a signal is used matters.** The same entity graph that predicts future combinations (§5.4) produced poor hypotheses when routed through candidate papers with a novelty filter (§5.4, §5.5).

**Judges must be validated.** An unvalidated small judge would have produced wrong effect sizes throughout.

**Disagreement is written into citations.** In CS abstracts, contradictions are rare between topically similar papers but appear where one paper explicitly revisits another; restricting candidates to citation links is what made the contradiction mode useful (§5.6).

**LLM-judge selection can overfit.** The hypothesis revision chosen by the 7B judge on dev queries did not beat the baseline for a human on held-out queries (§5.5), so human test sets remain necessary for generative modes.

## 7. Limitations

- Abstracts only; one corpus (CS, 2020–2025, 47,619 papers).
- Public citation benchmarks: the published comparison systems are from 2018, and our modern baselines are encoders (BGE, SPECTER2, SciNCL), not trained citation recommenders. The encoders' training data likely overlaps with DBLP and PubMed papers. The citation-graph benefit does not transfer fully to biomedicine (PubMed MRR drops).
- The GraphRAG baseline re-implements LightRAG's retrieval over our SciERC graph; the original LLM-built index (and Microsoft GraphRAG) was infeasible at this scale on our hardware, and a stronger LLM might narrow the gap.
- Project relevance labels come from an LLM judge (held-out κ 0.75 against the first annotator). On the other ten topics of the 44-query benchmark the judge agrees less with a human (κ 0.57, below our 0.6 bar) and is systematically strict; this does not bias graph comparisons but likely inflates the gain over the original system. Human–human agreement was measured on only 50 pairs (κ 0.91), with two annotators.
- Raters are anonymous and identified by role. Rater A, on every human task, is a project member who built the system, although every sheet was blind to method. Round 2 has only Rater A. Raters B and C are project members who did not build the system; their independence (no AI, no access to Rater A's sheets) was confirmed by the project owner but not recorded under a formal protocol.
- "Confirmed" in the time-sliced benchmark means later co-occurrence in abstracts, not a successful result. Entity extraction is automatic (F1 0.64).
- SciFact is biomedical; in-corpus contradiction precision rests on small human samples (68 pairs with two raters, 40 with one), and the round 2 headline on 9 pairs.
- SPECTER2 was evaluated without its query adapter.
- Three points deviate from our written protocol (EVALUATION_PROTOCOL §8): hypotheses were rated on three 1–5 scales instead of the five-dimension 1/3/5 rubric; the contradiction evaluation has at most 2 confirmed contradictions per arm, not the 10 positives per split the protocol asks for; and the judge-transfer check used one rater.
- The neuro-symbolic coupling is rule-based and hand-written (§3.1); rules such as the critical-language pattern were designed on this corpus.
- Hypotheses suggested as "never combined in the corpus" may be known in older literature.

## 8. Conclusion

Graph structure is not a general upgrade for scholarly RAG. It is redundant for topical ranking but adds substantial value for citation recommendation and research discovery. Evaluations of GraphRAG should name the task, use strong text baselines, and validate their judges.

## Reproducibility

Code, configurations, frozen query sets, judge prompts, and result artifacts are in the repository (`src/evaluation/*`, `results/*`). The full run is described in [RUN_FROM_SCRATCH.md](RUN_FROM_SCRATCH.md); `python -m src.evaluation.reproduce` recomputes 25 headline numbers from committed files. Data licences: Semantic Scholar abstracts (ODC-By), OpenAlex (CC0), SciFact (CC BY 4.0); LitSearch and SciDocs under their own terms.

## References

*Entries marked † were checked against publisher, proceedings or arXiv records on 2026-11-19 or 2026-11-20. The others are standard references written from known records; check each (authors, pages, DOI) and whether arXiv-only entries now have a peer-reviewed version before submission, then convert to the venue's citation style.*

- Adamic, L. A., & Adar, E. (2003). Friends and neighbors on the Web. *Social Networks*, 25(3), 211–230.
- Ajith, A., Xia, M., Chevalier, A., Goyal, T., Chen, D., & Gao, T. (2024). LitSearch: A retrieval benchmark for scientific literature search. In *Proceedings of EMNLP 2024*.
- Asai, A., et al. (2024). OpenScholar: Synthesizing scientific literature with retrieval-augmented LMs. arXiv:2411.14199.
- Beltagy, I., Lo, K., & Cohan, A. (2019). SciBERT: A pretrained language model for scientific text. In *Proceedings of EMNLP-IJCNLP 2019*.
- Bhagavatula, C., Feldman, S., Power, R., & Ammar, W. (2018). Content-based citation recommendation. In *Proceedings of NAACL-HLT 2018*.
- Cohan, A., Ammar, W., van Zuylen, M., & Cady, F. (2019). Structural scaffolds for citation intent classification in scientific publications. In *Proceedings of NAACL-HLT 2019*.
- Cohan, A., Feldman, S., Beltagy, I., Downey, D., & Weld, D. S. (2020). SPECTER: Document-level representation learning using citation-informed transformers. In *Proceedings of ACL 2020*.
- Cohen, J. (1968). Weighted kappa: Nominal scale agreement with provision for scaled disagreement or partial credit. *Psychological Bulletin*, 70(4), 213–220.
- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. In *Proceedings of SIGIR 2009*.
- Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. (2024). From local to global: A graph RAG approach to query-focused summarization. arXiv:2404.16130.
- Gao, T., Yen, H., Yu, J., & Chen, D. (2023). Enabling large language models to generate text with citations. In *Proceedings of EMNLP 2023*.
- †Garcez, A. d'Avila, & Lamb, L. C. (2023). Neurosymbolic AI: The 3rd wave. *Artificial Intelligence Review*, 56(11), 12387–12406. https://doi.org/10.1007/s10462-023-10448-w
- Guo, Z., Xia, L., Yu, Y., Ao, T., & Huang, C. (2024). LightRAG: Simple and fast retrieval-augmented generation. arXiv:2410.05779.
- †Gutiérrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., & Su, Y. (2024). HippoRAG: Neurobiologically inspired long-term memory for large language models. In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*.
- †Gutiérrez, B. J., Shu, Y., Qi, W., Zhou, S., & Su, Y. (2025). From RAG to memory: Non-parametric continual learning for large language models. In *Proceedings of ICML 2025*, PMLR 267, 21497–21515.
- †Han, H., et al. (2025). RAG vs. GraphRAG: A systematic evaluation and key insights. arXiv:2502.11371.
- He, P., Gao, J., & Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. In *Proceedings of ICLR 2023*.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- †Jurgens, D., Kumar, S., Hoover, R., McFarland, D., & Jurafsky, D. (2018). Measuring the evolution of a scientific field through citation frames. *Transactions of the ACL*, 6, 391–406.
- Kautz, H. (2022). The third AI summer: AAAI Robert S. Engelmore memorial lecture. *AI Magazine*, 43(1), 105–125. https://doi.org/10.1002/aaai.12036
- Kessler, M. M. (1963). Bibliographic coupling between scientific papers. *American Documentation*, 14(1), 10–25.
- Kinney, R., et al. (2023). The Semantic Scholar open data platform. arXiv:2301.10140.
- †Krenn, M., et al. (2023). Forecasting the future of artificial intelligence with machine learning-based link prediction in an exponentially growing knowledge network. *Nature Machine Intelligence*, 5, 1326–1335. https://doi.org/10.1038/s42256-023-00735-0
- Luan, Y., He, L., Ostendorf, M., & Hajishirzi, H. (2018). Multi-task identification of entities, relations, and coreference for scientific knowledge graph construction. In *Proceedings of EMNLP 2018*.
- †McNee, S. M., Albert, I., Cosley, D., Gopalkrishnan, P., Lam, S. K., Rashid, A. M., Konstan, J. A., & Riedl, J. (2002). On the recommending of citations for research papers. In *Proceedings of CSCW 2002*, 116–125. https://doi.org/10.1145/587078.587096
- †Nicholson, J. M., Mordaunt, M., Lopez, P., Uppala, A., Rosati, D., Rodrigues, N. P., Grabitz, P., & Rife, S. C. (2021). scite: A smart citation index that displays the context of citations and classifies their intent using deep learning. *Quantitative Science Studies*, 2(3), 882–898. https://doi.org/10.1162/qss_a_00146
- Ostendorff, M., Rethmeier, N., Augenstein, I., Gipp, B., & Rehm, G. (2022). Neighborhood contrastive learning for scientific document representations with citation embeddings. In *Proceedings of EMNLP 2022*.
- †Peng, B., Zhu, Y., Liu, Y., Bo, X., Shi, H., Hong, C., Zhang, Y., & Tang, S. (2024). Graph retrieval-augmented generation: A survey. arXiv:2408.08921.
- Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv:2205.01833.
- Qwen Team. (2024). Qwen2.5 technical report. arXiv:2412.15115.
- Ren, X., Liu, J., Yu, X., Khandelwal, U., Gu, Q., Wang, L., & Han, J. (2014). ClusCite: Effective citation recommendation by information network-based clustering. In *Proceedings of KDD 2014*.
- Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.
- †Si, C., Yang, D., & Hashimoto, T. (2025). Can LLMs generate novel research ideas? A large-scale human study with 100+ NLP researchers. In *Proceedings of ICLR 2025*.
- †Singh, A., D'Arcy, M., Cohan, A., Downey, D., & Feldman, S. (2023). SciRepEval: A multi-format benchmark for scientific document representations. In *Proceedings of EMNLP 2023*, 5548–5566.
- Skarlinski, M. D., et al. (2024). Language agents achieve superhuman synthesis of scientific knowledge. arXiv:2409.13740.
- Small, H. (1973). Co-citation in the scientific literature: A new measure of the relationship between two documents. *Journal of the American Society for Information Science*, 24(4), 265–269.
- †Strohman, T., Croft, W. B., & Jensen, D. (2007). Recommending citations for academic papers. In *Proceedings of SIGIR 2007*, 705–706. https://doi.org/10.1145/1277741.1277868
- Swanson, D. R. (1986). Fish oil, Raynaud's syndrome, and undiscovered public knowledge. *Perspectives in Biology and Medicine*, 30(1), 7–18.
- †Sybrandt, J., Tyagin, I., Shtutman, M., & Safro, I. (2020). AGATHA: Automatic graph mining and transformer based hypothesis generation approach. In *Proceedings of CIKM 2020*.
- Teufel, S., Siddharthan, A., & Tidhar, D. (2006). Automatic classification of citation function. In *Proceedings of EMNLP 2006*.
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In *NeurIPS 2021 Datasets and Benchmarks Track*.
- †Thomas, P., Spielman, S., Craswell, N., & Mitra, B. (2024). Large language models can accurately predict searcher preferences. In *Proceedings of SIGIR 2024*.
- †Upadhyay, S., Pradeep, R., Thakur, N., Craswell, N., & Lin, J. (2024). UMBRELA: UMbrela is the (open-source reproduction of the) Bing RELevance Assessor. arXiv:2406.06519.
- Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., & Hajishirzi, H. (2020). Fact or fiction: Verifying scientific claims. In *Proceedings of EMNLP 2020*.
- †Wang, Q., Downey, D., Ji, H., & Hope, T. (2024). SciMON: Scientific inspiration machines optimized for novelty. In *Proceedings of ACL 2024 (Volume 1: Long Papers)*, 279–299.
- †Xiang, Z., et al. (2026). When to use graphs in RAG: A comprehensive analysis for graph retrieval-augmented generation. In *Proceedings of ICLR 2026*.
- †Xiao, S., Liu, Z., Zhang, P., Muennighoff, N., Lian, D., & Nie, J.-Y. (2024). C-Pack: Packed resources for general Chinese embeddings. In *Proceedings of SIGIR 2024*. (BGE models.)
- Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In *NeurIPS 2023 Datasets and Benchmarks Track*.
