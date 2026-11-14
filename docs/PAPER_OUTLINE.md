# Paper Outline (draft)

Working title options:
- *Where Does Graph Structure Help Scholarly RAG? Not Topical Retrieval, but Citation Relatedness and Research Discovery*
- *Evaluating Provenance-Grounded GraphRAG for Scientific Literature: What Helps, What Does Not, and Why*
- *Grounded Literature Review, Contradiction Screening, and Hypothesis Generation over a Sparse Scientific Graph*

Framing: an **evaluated system plus empirical findings**, not a new model. Every claim below cites the artifact that supports it ([RESULTS.md](RESULTS.md)). Items marked ⏳ or ❌ must be completed before submission.

## 1. Contributions

1. **A provenance-grounded pipeline** for literature review, contradiction screening, and hypothesis generation over 47,619 CS papers, where every generated claim and every contradiction candidate is tied to verifiable sentence IDs, and unverifiable claims are withheld (structural citation validity 12/12 in end-to-end tests).
2. **A controlled retrieval study** on two public benchmarks (SciDocs, LitSearch) and two project benchmarks graded by a human-validated LLM judge (held-out κ 0.75 with a human annotator), with a frozen dev/test protocol:
   - general retriever + BM25 fusion beats SPECTER, SPECTER2 base, SciNCL, and OpenScholar (all p < 0.001 on LitSearch);
   - on 44 independent topic-stratified queries, NDCG@10 0.644 vs 0.287 for the original system (+0.358, 95% CI [0.285, 0.434]); the gain comes from the embedding and BM25 fusion, while graph expansion adds a non-significant +0.023.
   - **graph signals predict future research:** in a time-sliced benchmark (pre-2023 data, 120 tasks), graph-based method–task scores nearly double average precision over text similarity (0.083 vs 0.044, p < 0.0001; robust to cutoff and threshold);
   - **where the graph helps:** on citation recommendation (400 test papers, real citations as ground truth, leakage-controlled), co-citation over the densified graph adds +0.068 Recall@20 over BGE + BM25 (28% relative, p < 0.0001), while for topical queries the densified graph adds nothing (§2e).
3. **Negative and diagnostic findings** that matter for GraphRAG on scholarly data:
   - a small LLM relevance judge (3B) produced batch-position-biased labels (κ 0.17 with a human); validating the judge on human grades changed the measured effect sizes, and one paper per call fixed a 7B judge (κ 0.60 → 0.77);
   - concept-overlap structural holes produce *less* relevant hypothesis candidates than nearest neighbours (human rating 3.27 vs 4.20 of 5, p = 0.002); a problem-bridge, novel-method revision is under test;
   - NLI contradiction screening over abstracts finds no real contradictions in 56 top-ranked pairs (human labels); a citation-grounded index and a verdict filter (4.8% false alarms) are under test;
   - reference lists were missing for 77% of papers; filling them from Semantic Scholar and OpenAlex cut isolated papers from 71% to 2%;
   - citation expansion fails when only 3.6% of references resolve in-corpus; semantic kNN edges turn graph expansion from harmful to helpful;
   - noun-phrase concept graphs manufacture contradiction candidates and collapse novelty scores (HNS constant at 0.333);
   - a dev-selected configuration that did not generalize, reported with the protocol that caught it.
4. **Component validation on expert labels:** SciERC NER (fine-tuned SciBERT, typed F1 0.639 vs 0.240 zero-shot GLiNER), SciFact contradiction screening (NLI AUROC 0.818), and verdict models of 3B–120B parameters (F1 0.53–0.81; a local 7B reaches 0.70).

## 2. Related work (positioning)

| Line of work | Representative | How this paper differs |
|---|---|---|
| Graph-based RAG | Microsoft GraphRAG (Edge et al., 2024): LLM-built entity graphs with community summaries | Uses citation and similarity structure plus a supervised scientific entity extractor; measures when the graph helps rather than assuming it does |
| Scientific RAG systems | OpenScholar (Asai et al., 2024): retrieval over ~45M open-access papers, ScholarQABench | Smaller, fully local pipeline; emphasis on sentence-level provenance and controlled ablations rather than end-task win rate |
| Agentic literature synthesis **with contradiction detection** | PaperQA2 (Skarlinski et al., 2024): reports ~2.3 contradictions per biology paper, 70% validated by experts | Closest prior work for the contradiction task. Differences to argue: abstract-level CS corpus, explicit candidate screening with sentence evidence, and component evaluation on expert labels (SciFact). A direct comparison or a clear scope argument is required |
| Attributed generation evaluation | ALCE (Gao et al., 2023) | Adopts citation precision and NLI support; adds contradiction screening with evidence |
| Literature search benchmarks | SciDocs (Cohan et al., 2020; BEIR, Thakur et al., 2021); LitSearch (Ajith et al., 2024) | External validation. SciDocs BM25 and MiniLM scores match published BEIR values; LitSearch reproduces the reported direction (dense retrievers beat BM25), not specific published numbers |
| Scientific claim verification | SciFact (Wadden et al., 2020) | Used to validate the contradiction components on expert labels |
| Scientific IE | SciERC (Luan et al., 2018) | Used to select the entity extractor that builds the concept graph |
| Hypothesis generation | literature-based discovery; LLM hypothesis generation surveys | Structural-hole candidates from typed entities; ❌ human rubric evaluation pending |

## 3. Method

1. Corpus: Semantic Scholar, 11 CS topics, 2020–2025, 47,619 abstracts; Neo4j graph (papers, authors, citations, `SIMILAR_TO`, `SciEntity`); Chroma vectors.
2. Retrieval: BGE dense + BM25 + graph expansion (citations and kNN edges, relative gate), weighted RRF.
3. Review: sentence passages with stable IDs; structured claim/evidence generation; invalid citations withheld; optional NLI support check.
4. Contradiction: finding-sentence filter → symmetric NLI scoring → structured LLM verdict with the two evidence sentences.
5. Hypothesis: typed-entity structural holes ranked by evidence; LLM generation with feasibility and missing evidence.

## 4. Experiments and results map

| Claim | Evidence | Status |
|---|---|---|
| Retrieval comparison on public benchmarks | RESULTS §1 | ✅ |
| Retrieval on project benchmarks, validated labels | RESULTS §2c, §2d | ✅ (one human validated the judge) |
| AI relevance judge agrees with humans | RESULTS §2b | ✅ one annotator; ⏳ second annotator |
| NER selection | RESULTS §3 | ✅ |
| Concept graph quality | RESULTS §4 | ✅ (probe) |
| Hypothesis signals, time-sliced discovery benchmark | RESULTS §4c | ✅ positive, robust |
| Hypothesis quality, human ratings | RESULTS §4b | ✅ round 1 (negative); ⏳ round 2 (revised method, held-out queries) |
| Contradiction screening and verdicts | RESULTS §5, §5b, §5b-2 | ✅ SciFact verdicts; ✅ round 1 in-domain (negative); ⏳ round 2 index |
| Answer grounding: vector RAG vs GraphRAG | RESULTS §5c | ✅ (no significant difference) |
| Citation densification | RESULTS §6 | ✅ |
| Graph contribution: topical retrieval vs citation recommendation | RESULTS §2e, §2f | ✅ |
| Metric implementation correctness | `tests/test_ir_metrics_vs_trec_eval.py` (exact match to trec_eval) | ✅ |
| Scaling with corpus size | planned citation-closure study | ❌ future work or extension |

## 5. Limitations to state explicitly

- Project relevance labels come from an LLM judge validated against one human annotator (held-out κ 0.75); a second annotator is pending.
- SciFact is biomedical; contradiction components are validated off-domain.
- SPECTER2 was evaluated without its query adapter (dependency conflict); its scores are a lower bound.
- Abstracts only; no full text.
- Local 3B generator; verdict confidence is uncalibrated.

## 6. Target venues

- Workshops: Scholarly Document Processing (SDP, at ACL/EMNLP), AI-for-science and research-assistant workshops, ACL Student Research Workshop.
- Main conferences require the human studies, the stratified benchmark, and a stronger novel component (for example the scaling study or a fine-tuned local model).

## 7. Submission checklist

- [x] Human relevance sample (200 pairs, 1 annotator) and judge validation
- [ ] Second relevance annotator (human–human κ)
- [x] Human hypothesis ratings, round 1 (1 rater)
- [ ] Hypothesis and contradiction round 2, two raters, weighted kappa
- [x] Stratified 44-query benchmark pooled and graded
- [x] Contradiction verdict comparison completed (3B, 7B, 27B, 120B)
- [x] Vector RAG vs GraphRAG grounding comparison completed
- [ ] Bibliography verified against the original papers
- [ ] Code and data release plan (licenses: SciFact CC BY 4.0, abstracts ODC-By, LitSearch and SciDocs terms)

## 8. References verified (2026-11-14)

Checked against arXiv, ACL Anthology, or publisher records:

- Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, arXiv:2404.16130 (2024).
- Asai et al., *OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs*, arXiv:2411.14199 (2024).
- Skarlinski et al., *Language agents achieve superhuman synthesis of scientific knowledge* (PaperQA2), arXiv:2409.13740 (2024).
- Gao, Yen, Yu, Chen, *Enabling Large Language Models to Generate Text with Citations* (ALCE), EMNLP 2023, pp. 6465–6488.
- Ajith et al., *LitSearch: A Retrieval Benchmark for Scientific Literature Search*, EMNLP 2024, pp. 15068–15083; arXiv:2407.18940.
- Ostendorff et al., *Neighborhood Contrastive Learning for Scientific Document Representations with Citation Embeddings* (SciNCL), EMNLP 2022, pp. 11670–11688.

Still to verify before submission: SPECTER (Cohan et al., ACL 2020), SPECTER2/SciRepEval (Singh et al., 2023), BEIR (Thakur et al., NeurIPS 2021 Datasets and Benchmarks), SciFact (Wadden et al., EMNLP 2020), SciERC (Luan et al., EMNLP 2018), BGE/C-Pack (Xiao et al., 2023), GLiNER (Zaratiana et al., NAACL 2024), condensed-list evaluation (Sakai, SIGIR 2007).
