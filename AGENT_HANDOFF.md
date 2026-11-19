# NeSy-GraphRAG Continuation Handoff

Last reconciled: November 19, 2026. Read this file before changing code, data, configuration, or documentation. It is the canonical handoff for a new chat or coding agent. `PROJECT_STATUS.md` is the concise status view; [docs/RUN_FROM_SCRATCH.md](docs/RUN_FROM_SCRATCH.md) is the install, build and run guide (it replaces the old `SETUP.md` content); `LOCAL_PROJECT_RUNBOOK.md` is a local-only operations reference.

**Current stage (November 19):** research and evaluation are complete, and `master` holds everything. Read the November 19 section first; the older sections below are kept as the record of how each decision was made.

## Mission and Scope

The Phase 1 report defines the project as a trustworthy research assistant that:

- synthesizes scientific literature with traceable evidence;
- reasons across papers to surface possible contradictions;
- generates evidence-based research hypotheses; and
- evaluates reliability rather than treating fluent LLM output as scientific truth.

The Phase 1 report (August-December 2025) listed noisy-PDF extraction and model training as Phase 2 objectives. The team later deferred both, so they are not part of the current scope.

The current implementation is an abstract-and-metadata prototype. It is deliberately **not** a PDF/full-text ingestion system, a trained/fine-tuned model, a fully autonomous research agent, or a source of validated scientific conclusions. Do not add PDF ingestion unless the user explicitly reopens that scope.

## November 19, 2026: Current State

**Branches.** `master` contains all work. `research-grade-evaluation` (tags `paper-freeze-2026-11-19`, `app-freeze-2026-11-19`) is the paper state before the two new baselines; `baselines-graphrag-citation` adds them; `corpus-expansion-75k` is an untouched placeholder for a 75k-paper corpus.

**Stores** (unchanged since November 15): 47,619 papers in clean, NER, Chroma (`s2_papers_bge`, plus the historical `s2_papers` and `s2_papers_specter_ta`) and Neo4j. Neo4j has `CITES` 22,370, `CITES_ADDED` 133,527, `COUPLED_WITH` 411,418, `SIMILAR_TO` 238,095, `SciEntity` 323,624 with `MENTIONS` 696,390, and `ExternalPaper` 24,714.

**Application** (`app/streamlit_app.py`; configuration in the local `.env`, mirrored in `.env.example`):
- Four modes: Literature Review, Contradiction Detection (`CONTRADICTION_CANDIDATE_MODE=citation`), Hypothesis Generation (`HYPOTHESIS_GENERATOR=ranker`), and Related Work.
- Related Work shows an interactive citation map (`app/graph_view.py`, vis-network 9.1.9 loaded from unpkg) and lists every citation behind each suggestion (`related_work()` returns `graph`, `cited_by_similar`, `cites` and `cited_by`).
- Review shows a red error with the reason when the LLM returns nothing (usually Ollama refusing for lack of free RAM), instead of empty 0/0 checks.
- `OLLAMA_KEEP_ALIVE=30m` keeps the model loaded between queries.
- On a 16 GB machine, run only one model-heavy job at a time.

**Evaluation added November 19** (RESULTS §2g–2h):
- **LightRAG-style baseline** (`src/evaluation/lightrag_baseline.py`; stages `build`, `keywords`, `rank`, `grade`, `score`; index under ignored `data/lightrag_style/`; results in `results/retrieval/stratified_v2_lightrag_20261119/`). It runs LightRAG's query algorithm over the SciERC graph. Mix 0.396 and hybrid 0.176 NDCG@10, vs 0.642 for this system. The frozen stratified directory is only read.
- **DBLP/PubMed citation recommendation** (`src/evaluation/citeomatic_benchmark.py`, citeomatic protocol, no training; data: the two `comparison/*/corpus.json` files under ignored `data/external/citeomatic/`; results in `results/retrieval/citeomatic_20261119/`).
  - DBLP test: F1@20 0.284 and MRR 0.677, vs 0.230 and 0.649 for text alone and 0.303 and 0.689 for trained Citeomatic.
  - PubMed test: recall rises but MRR falls (0.699 vs 0.747). `src/evaluation/citeomatic_top1_diagnosis.py` shows why: text alone is right at rank 1 71% of the time there, vs 45% for the graph.
- **Human ratings complete.**
  - Second relevance grader on a blinded 50-pair subset: κ 0.91.
  - Second rater for round-1 hypotheses (κ 0.82–0.90) and contradictions (κ 0.76).
  - Round 2 (rater 1): hypotheses (the bridge does not beat vector neighbours) and contradictions (56% tension-or-contradiction precision for verdict-flagged pairs).
  - Template/AI-generated second-rater sheets are kept in `evaluation/human_review/excluded/` and used nowhere.
- **Citation-trained encoders** (`citeomatic_benchmark --encoder specter2_base|scincl`). Co-citation raises Recall@20 on top of all three encoders in both domains. SPECTER2 + BM25 + co-citation on DBLP test: 0.300 / 0.708 vs trained Citeomatic 0.303 / 0.689. The script exits with `os._exit` after writing results, because Windows CUDA teardown hung.
- `src/evaluation/reproduce.py` checks 23 headline numbers. 406 tests pass.

**Review fixes (November 19, on `baselines-graphrag-citation`).** An external-style review listed six blockers. Done:
- `reproduce.py` no longer needs the gitignored subset key; `build_relevance_subset.subset_mapping()` rebuilds it deterministically. Verified 23/23 in a fresh clone.
- Overstatements corrected in the abstract, README, RESULTS and PROJECT_STATUS: "Selected vs BGE alone" is borderline (p = 0.0095, Holm-adjusted 0.048), not plainly significant; SPECTER2 is citation-pretrained, so "untrained … matches Citeomatic" became "no training on DBLP … comes close to"; contradiction precision now carries counts and CIs (5/9, 27–81%) and the single-rater caveat; "exactly as relevant" became "about as relevant"; the abstract lists four findings and four public benchmarks.
- Related work rewritten with the closest prior work (Strohman et al. 2007; McNee et al. 2002; Citeomatic NNSelect; Small; Kessler; citation function; AGATHA, Science4Cast, SciMON; HippoRAG; LLM-judge papers) and a 41-entry reference list (11 checked online, marked †).
- PAPER_DRAFT §3.1 explains what is neural and what is symbolic.
- `evaluation/human_review/build_judge_transfer_sheet.py` and `judge_transfer_sheet_blank.csv`: 100 blinded pairs from the ten non-GNN topics to check the judge (task 9). The key is gitignored and also rebuildable.
- `evaluation/human_review/README.md` has a Rater record table with blanks for the owner.

Still needs people: grading task 9; filling the rater record (who rater 3 is, whether they were involved in development); a second rater for round 2.

**Local `.env` warning.** On November 19 the owner's `.env` had `NEO4J_ALLOW_RESET=true`. It should be `false` except during an intentional graph rebuild.

## Architecture Implemented

1. Semantic Scholar multi-topic ingestion is resumable, deduplicates by paper ID, and preserves prior records/checkpoints.
2. Cleaning produces a normalized paper corpus; spaCy NER extracts authors/entities/noun chunks.
3. SPECTER embeddings are persisted in ChromaDB. CUDA, MPS, or CPU is selected automatically where available.
4. Neo4j stores `Paper`, `Author`, and `Concept` nodes plus authorship, concept, and real citation edges.
5. Retrieval starts with Chroma semantic seeds, expands through Neo4j citations, filters weak graph neighbours by semantic similarity/query coverage/connectivity, then applies weighted reciprocal-rank fusion.
6. Review generation uses the provider-neutral client: local Ollama by default, optional Groq. Retrieved abstracts are split into stable sentence passages; invalid or missing passage IDs cause claims to be rejected or audited.
7. Contradiction mode ranks cross-paper candidates using shared concepts and temporal distance, then requires a structured LLM verdict.
8. Hypothesis mode ranks graph-evidence candidates and retains feasibility/support/missing-evidence fields.
9. Streamlit exposes all three workflows. Evaluation utilities cover BM25, vector-only, graph-only, evaluation hybrid, production fusion, reasoning benchmarks, and artifact auditing.

## Current Verified State

The following was checked on November 1, 2026 from the current working directory:

| Asset | Current value | Meaning |
|---|---:|---|
| Raw Semantic Scholar records | 52,822 | Durable merged ingestion output |
| Clean papers | 47,619 | Retrieval/indexing source corpus |
| NER records | 47,619 | No NER backlog in the current data files |
| Chroma collection `s2_papers` | 47,619 vectors | Matches clean corpus count |
| Neo4j `Paper` nodes | 47,619 | Matches clean corpus count |
| Neo4j `CITES` edges | 22,370 | Sparse citation graph; not a failure |
| Neo4j Bolt port | Available on `127.0.0.1:7687` | Live graph service at verification time |
| Ollama port | Not running on `127.0.0.1:11434` | Start it before generation/evaluation |
| Streamlit port | Not running on `127.0.0.1:8501` | Start it for the UI |

The current branch is `phase3`, at commit `8f2eaa2`. It is four commits ahead of `origin/phase3` and matches `origin/feat/local-evaluation`. Do not assume the four commits are already pushed to `origin/phase3`.

### Corpus Snapshot Mismatch

The frozen expanded retrieval pool in `results/retrieval/expanded_20260913/` was prepared against a **47,627-paper** corpus (clean-file SHA-256 `85b3783e...`, the September 13 fetch), not the current **47,619-paper** snapshot (SHA-256 `dc3c6d30...`). Resuming `grade` and running `evaluate` are unaffected because they use only the saved candidates and rankings, but those scores describe the 47,627-paper corpus. `matched_rag` retrieves live from the current stores, so its outputs belong to the 47,619-paper corpus. Label each result with its corpus and do not present the two as one snapshot.

### Test Record

The latest complete test record is **334 passing tests** (November 13, 2026; 306 earlier tests plus 28 for the new modules). The earlier record was **306 passing tests** from the September 14 cleanup checkpoint. The current data/store parity above was rechecked on November 1; the full suite was not rerun as part of this documentation reconciliation.

## What Has Been Executed

- Multi-topic CSE ingestion, cleaning, NER, Chroma indexing, and Neo4j loading completed for the 47,619-paper corpus.
- Store audit previously found no missing, duplicate, or mismatched paper IDs across clean/NER/Chroma/Neo4j. The current counts still match.
- Local Ollama provider support is implemented across review, contradiction, hypothesis, Streamlit, and general evaluation paths. Groq remains optional.
- Sentence-level claim provenance, citation validation, structured contradiction parsing, and evidence-ranked hypothesis validation are implemented and tested.
- BM25, vector-only, graph-only, historical hybrid, production-fusion retrieval paths, frozen candidate pools, checkpointed relevance grading, SciFact evaluation, support calibration, and packaged reasoning reproduction are implemented.
- Broad-corpus retrieval grading completed on November 13, 2026 (**1,145 of 1,145**); see the November 13 section for scores. `PAUSED.json` in that directory is a historical record of the September pause.
- A prior 8,850-paper benchmark reported NDCG@10 of 0.3676 for evaluation hybrid versus 0.3643 for vector-only, with paired randomization `p=1.0`. This is historical context only, not evidence of a current expanded-corpus improvement.
- The separate SciFact NLI check reported contradiction F1 0.6286 and two-class macro F1 0.5143. It assesses claim/evidence classification in biomedical abstracts, not graph retrieval or cross-paper scientific contradiction discovery.

## November 13, 2026 Work Session

### Retrieval benchmark completed

The paused grading finished: **1,145 of 1,145** AI-reference labels (grades: 541 zero, 126 one, 478 two). The frozen pool was scored with `refresh_retrieval evaluate` (`results/retrieval/expanded_20260913/{dev,test}/`). These scores describe the 47,627-paper pool corpus.

| Test, 14 queries | NDCG@10 | P@10 | Recall@20 | MAP |
|---|---:|---:|---:|---:|
| BM25 | 0.529 | 0.629 | 0.414 | 0.289 |
| production (graph-filtered fusion) | 0.455 | 0.507 | 0.341 | 0.220 |
| vector (SPECTER, abstract) | 0.435 | 0.507 | 0.341 | 0.215 |
| evaluation hybrid (16:1) | 0.435 | 0.507 | 0.341 | 0.215 |
| graph-only | 0.413 | 0.514 | 0.314 | 0.192 |

BM25 vs vector: Δ +0.094 NDCG@10, 95% CI [−0.037, +0.220], p = 0.19. Production vs vector: Δ +0.020, p = 0.50. No difference is significant with 14 queries.

### Diagnosis (measured on the live stores)

- Only 22,370 of 625,222 references (3.6%) resolve inside the corpus; 71.4% of papers have no citation edge. Under the absolute gate, a graph paper reached the top 10 for 3 of 20 benchmark queries and 1 of 11 topic queries.
- 83.5% of concepts occur in one paper; "this paper" links 7,013 papers. Before enrichment, 28 of 29 random paper pairs had concept-path length 2, so HNS was ≈ 0.33 for nearly any hypothesis, and the hypothesis query's no-citation clause removed 2 of 7,313 candidates.
- The historical Chroma index embeds lowercased abstracts without titles; SPECTER was trained on `title [SEP] abstract`.

### Changes (all opt-in; defaults reproduce the frozen protocols)

| Change | Where | Setting |
|---|---|---|
| Generic-concept flags (`Concept.df`, `generic`, `generic_reason`): 32,789 concepts, 136,972 of 475,121 concept edges | `src/storage/graph_enrichment.py concepts --write` (applied) | `CONCEPT_EXCLUDE_GENERIC=true` (default false) |
| `SIMILAR_TO` kNN edges, k = 5 from `s2_papers` vectors: 238,095 edges, cosine 0.73–1.00, median 0.907 | `graph_enrichment similarity --write` (applied) | `GRAPH_EXPANSION_EDGES=CITES\|SIMILAR_TO` |
| Title+abstract SPECTER index in new collection `s2_papers_specter_ta` (47,619 vectors; `s2_papers` untouched) | `chroma_store` | `EMBEDDING_TEXT_MODE=title_abstract`, `CHROMA_COLLECTION=s2_papers_specter_ta` |
| BM25 as a third RRF arm | `retrieval.fuse_results` | `BM25_FUSION_WEIGHT` (default 0) |
| Relative graph gate and configurable candidate limit | `retrieval.filter_symbolic_candidates` | `GRAPH_GATE_MODE=relative`, `GRAPH_RELATIVE_MARGIN`, `GRAPH_CANDIDATE_LIMIT` |
| Schema-constrained JSON contradiction verdicts (Ollama); legacy format still parsed | `prompts`, `verdicts`, `contradiction` | `CONTRADICTION_STRUCTURED_OUTPUT=true` (default) |
| Contradiction year term and same-year exclusion made explicit | `contradiction` | `CONTRADICTION_YEAR_WEIGHT` (0.25), `CONTRADICTION_REQUIRE_DIFFERENT_YEAR` (true) |
| Structural-hole test can also exclude semantically linked candidates | `hypothesis` | `HYPOTHESIS_HOLE_EDGES=CITES\|SIMILAR_TO` |
| HNS restricted to concept links, skipping generic concepts via an inline path predicate (the earlier all-nodes filter took minutes per hypothesis) | `metrics.compute_hns` | follows `CONCEPT_EXCLUDE_GENERIC` |
| Frozen-pool hash check tolerates CRLF/LF checkout differences only | `refresh_retrieval.verify_pool` | — |
| Dev ablation with standard and condensed scoring; pool extension with the same blinded judge | `src/evaluation/variant_ablation.py`, `src/evaluation/pool_extension.py` | — |
| Topic-stratified 44-query draft (2 dev + 2 test per topic), pending team review | `evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json` | `refresh_retrieval prepare --benchmark` |
| Evaluation protocol | `docs/EVALUATION_PROTOCOL.md` | — |

Undo the graph enrichment with `MATCH ()-[r:SIMILAR_TO]->() DELETE r` and `MATCH (c:Concept) REMOVE c.df, c.generic, c.generic_reason`.

### Variant selection on dev, confirmed once on test

Dev (6 queries, pool extended by 113 labels from the same judge) selected `full_ta_bm25_similar` (title+abstract index, BM25 arm, CITES+SIMILAR_TO expansion with relative gate): NDCG@10 0.548 vs production 0.458. The selection was frozen in `results/retrieval/variant_ablation_dev_20261113/selection.json` before test was opened.

Test (14 queries, pool extended by 82 labels): selected 0.476, production 0.449 (Δ +0.027, p = 0.56), **BM25 0.529** (selected − BM25 = −0.053, p = 0.41). The dev gain did not generalize, so production defaults were not changed. On the dev split, similarity edges turned graph expansion from harmful (citations only, −0.065 vs production) into neutral-to-positive (+0.027).

### Reasoning candidates after generic-concept exclusion

Measured over the 20 benchmark and 6 topic queries (`results/local/reasoning_probe_after_enrichment_20261113.json`): contradiction candidates fall from a mean of 2.4 per query to **0 for all 26 queries**, so the concept-overlap generator depended entirely on generic hub phrases and needs replacing (for example NLI over co-retrieved sentence pairs). Hypothesis candidates vary from 0 to 25 per query (mean 19.2) with specific shared concepts ("fraud detection", "shap", "message-passing gnns"). HNS on 12 random pairs takes values 0.33 and 0.67 instead of a near-constant 0.33. This is why `CONCEPT_EXCLUDE_GENERIC` defaults to false.

### Data quality

51 near-duplicate paper pairs (cosine ≥ 0.995), mostly preprint/published versions under different IDs. Deduplication is pending.

## November 14, 2026: Research-Model Selection and Deployment

Full result tables: [docs/RESULTS.md](docs/RESULTS.md).

**Active configuration** (set in the local `.env`; the original was backed up outside the repository):

```dotenv
CHROMA_COLLECTION=s2_papers_bge
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_TEXT_MODE=title_abstract_plain
EMBEDDING_QUERY_PREFIX=Represent this sentence for searching relevant passages:
BM25_FUSION_WEIGHT=1.0
GRAPH_GATE_MODE=relative
GRAPH_EXPANSION_EDGES=CITES|SIMILAR_TO
GRAPH_CANDIDATE_LIMIT=50
CONCEPT_SOURCE=scientific
CONCEPT_EXCLUDE_GENERIC=true
CONTRADICTION_CANDIDATE_MODE=nli
OLLAMA_KEEP_ALIVE=2m
```

| Component | Choice | Evidence |
|---|---|---|
| Retrieval embedding | BGE-base-en-v1.5 (`s2_papers_bge`, 47,619 vectors) | Beats SPECTER, SPECTER2 base, SciNCL, and OpenScholar on SciDocs and LitSearch (all p < 0.001); 44 stratified queries with validated labels: NDCG@10 0.644 vs 0.287 for the previous production (RESULTS §2d) |
| Retrieval fusion | BGE + BM25 + graph (CITES and SIMILAR_TO, relative gate) | Dev-selected, test-confirmed once |
| Entity extraction | SciBERT fine-tuned on SciERC (`models/scibert-scierc-ner`, gitignored) | SciERC test typed F1 0.639 vs GLiNER 0.240 and community checkpoints ≤ 0.294; spaCy span F1 0.270 |
| Concept graph | `(:Paper)-[:MENTIONS]->(:SciEntity {name, type, df, generic})`: 323,624 nodes, 696,390 edges | Shared concepts become task/method terms instead of discourse phrases |
| Contradiction candidates | NLI over authors' finding sentences (`src/pipeline/contradiction_nli.py`) | Concept overlap yields few or noise-driven pairs; NLI returns sentence evidence |

Rebuild order for the scientific concept graph: `python -m src.ingestion.train_scierc_ner` (needs `data/external/zj88zj__SCIERC/{train,dev,test}.json`), then `python -m src.ingestion.scientific_ner extract --model models/scibert-scierc-ner`, then `python -m src.storage.graph_enrichment scientific-entities --write`. The historical spaCy `Concept` graph remains in Neo4j; set `CONCEPT_SOURCE=spacy` to use it. Remove the new graph with `MATCH (e:SciEntity) CALL (e) { DETACH DELETE e } IN TRANSACTIONS OF 5000 ROWS` (run in an auto-commit session).

End-to-end smoke test (`results/local/smoke_test_20261114.log`): review 6/6 claims grounded with 12/12 valid citations from neural, lexical, symbolic, and overlapping sources; contradiction 5/5 valid structured verdicts; hypothesis 5 accepted. The local 3B model reports confidence 1.0 for every verdict, so verdict confidence is uncalibrated.

**Environment notes.** `gliner`, `sentencepiece`, and `protobuf==7.36.1` were added (protobuf must be at least the gencode version sentencepiece was built with). The `adapters` package was not installed because it would downgrade `transformers`. The local environment runs spaCy 3.7.5 although `requirements.txt` pins 3.8.16; this predates the November work. Hugging Face's Xet client hung on some downloads; `HF_HUB_DISABLE_XET=1` works around it. Unit tests pin all selectable options in `tests/conftest.py`, so they do not depend on `.env`.

## November 14–15, 2026: Human Validation, Stratified Benchmark, Densification

Full tables: [docs/RESULTS.md](docs/RESULTS.md) §2b–2d, §4b, §5b-2, §6. Human tasks: [evaluation/human_review/README.md](evaluation/human_review/README.md).

**Relevance judge.** `src/evaluation/judge_agreement.py` splits the human sheet (`relevance_sheet_cv.csv`, 200 pairs, annotator CV) 50/50 by hash; judges are chosen on the selection half and scored once on the held-out half. Frozen 3B pool labels: κ 0.17 (batch-position bias; stored labels match raw outputs). Local `qwen2.5:7b-instruct`, strict prompt, `--batch-size 1`: held-out κ 0.75 (**used for grading**). Groq `qwen/qwen3.8-27b`, strict: held-out κ 0.88, but the free tier (200K tokens/day) cannot grade full pools. Do not use additional Groq keys to bypass rate limits.

**Retrieval with validated labels.**
- `src/evaluation/regrade_pool.py` regrades the 1,644 GNN-pool pairs into `results/retrieval/regraded_qwen2_5_7b_strict_b1_20261114/`: selected 0.813 vs production 0.349 on test (14/14 wins).
- `src/evaluation/stratified_benchmark.py` (`rank --family specter|bge`, `grade`, `score`) evaluates the reviewed 44-query set v0.5 (`results/retrieval/stratified_v2_20261114/`): selected 0.644 vs production 0.287. The selected system is not significantly better than BM25 (+0.019) or BGE + BM25 (+0.023). The query file is hash-locked once pooling starts.

**Hypotheses.** Round 1 (two raters, 30 queries) rated structural holes below a vector baseline. Those candidates ran with `CONCEPT_EXCLUDE_GENERIC=false`, and 21 of 30 differ with the filter on. `src/pipeline/hypothesis_bridge.py` (problem bridge plus a novel method) was selected on the same queries with the 7B judge (`src/evaluation/hypothesis_dev.py`; relevance 0.43 → 0.77). Round 2 (`build_round2_kit.py`, 14 held-out queries, 4 blinded arms, generic filter on; rater 1, `hypothesis_round2_report_cv.json`): the bridge did not beat vector neighbours (relevance 3.54 vs 3.93, n.s.), so the application keeps the method–task ranker, not the bridge.

**Contradictions.** Round 1 (two raters, κ 0.76): no pair of the 56 top NLI pairs is a contradiction for both raters; rater 2 labels 2 (C013, C021), rater 1 calls them tensions. A 7B verdict filter (`src/evaluation/contradiction_human_eval.py`) flags 5 of 68, including both disputed pairs (3 false alarms, about 5%; SciFact F1 0.70). `src/pipeline/contradiction_citation.py` and `src/evaluation/contradiction_index.py` build a corpus-wide citation-grounded index (591 candidates; 9 of the top 150 flagged). Round 2 (rater 1, `contradiction_round2_report_cv.json`): 5 of 9 verdict-flagged index pairs are a contradiction or tension (2 contradictions), so the application's citation mode is supported.

**Densification.** `src/ingestion/citation_densify.py` (`fetch-s2`, `fetch-openalex`, `build`, `fetch-external`, `write --write`, `stats`) has written additive layers: `CITES_ADDED` 133,527, `COUPLED_WITH` 411,418, `ExternalPaper` 24,714 with `CITES_EXTERNAL` 641,616. Inputs are in ignored `data/` files (`s2_references_supplement.json`, `openalex_references.json`, `openalex_external_papers.json`, `citation_densify_plan.json`). `OPENALEX_API_KEY` goes in `.env`. The new edge types are allowed in `GRAPH_EXPANSION_EDGES` but are not enabled by default. Undo: delete those relationship types and `ExternalPaper` nodes in auto-commit batches.

**Graph contribution.** `src/evaluation/dense_graph_eval.py` (dev select, test once): the dense layers do not raise topical NDCG (papers they add are as relevant as the ones they displace). `src/evaluation/citation_recommendation.py`: co-citation over the dense graph adds +0.068 Recall@20 over BGE + BM25 on 400 test papers. `COLLAPSE_NEAR_DUPLICATES=true` is set in `.env`; no data is deleted.

**Application switched to evidence-backed methods (2026-11-15; reversible in `.env`, backup outside the repository).**
- `OLLAMA_MODEL=qwen2.5:7b-instruct`: the 3B default had verdict F1 0.53 and judge κ 0.44; the 7B has 0.70 and 0.75.
- `CONTRADICTION_CANDIDATE_MODE=citation`: pairs from the corpus-wide citation index (`CONTRADICTION_INDEX_FILE`) touching the query's top 50 papers, re-verdicted by the app LLM. 27 of the 44 stratified queries get at least one pair. The previous value was `nli`.
- `HYPOTHESIS_GENERATOR=ranker`: method–task pairs from `src/pipeline/hypothesis_ranker.py` (the §4c scorer; time-sliced precision@10 0.067 vs 0.038 for the vector baseline and 0.021 for the `bridge` generator, RESULTS §4d). `bridge` and `structural_hole` remain available. The ranker is not yet human-rated.
- A **Related Work** mode (`src/pipeline/related_work.py`) uses the citation-recommendation configuration.
- Both reasoning switches await round 2 human confirmation. Code defaults are unchanged (`concept`, `structural_hole`, 3B), and the tests pin them.

**Environment.** `.env` was overwritten on Nov 14 at 22:14 by an older copy; the BGE settings and `CONCEPT_EXCLUDE_GENERIC=true` were restored (pre-restore copy outside the repository). `ollama pull qwen2.5:7b-instruct` is required for the judge and verdict filter. The S2 client now backs off exponentially on 429.

## Important Limitations

- Citation expansion is sparse. More papers broaden semantic retrieval but do not automatically create useful graph contribution.
- spaCy noun chunks are noisy concept proxies; concept density is not concept quality.
- Retrieval judgments are LLM-judge labels validated against one human annotator (held-out κ 0.75), not human ground truth.
- The old 20-query set concerns graph-neural-network topics; the 44-query stratified set (v0.5) covers all 11 topics.
- The frozen in-domain contradiction package has inadequate positive-class coverage; do not make strong contradiction-quality claims from it.
- Claim provenance proves that a cited passage exists and is structurally valid. It does not by itself prove full semantic entailment.
- Hypotheses, contradictions, and reviews are model outputs requiring human/expert review.

## Immediate Next Work

Continue in this order unless the user changes priorities:

1. Format `docs/PAPER_DRAFT.md` for the venue the user chooses (template, page limit, figures, AI-use and data-availability statements). Check the references not marked †.
2. When the task 9 sheet comes back, score it (`build_judge_transfer_sheet.py score`), add the κ to RESULTS §2b and PAPER_DRAFT §4.3/§7, and add it to `reproduce.py`. If κ < 0.6, say the 44-query numbers rest on an unvalidated judge for those topics.
3. Optional evaluation:
   - a second rater for round 2;
   - more labelled contradiction pairs;
   - human ratings of the method–task ranker.
4. Optional: the 75k-paper corpus on `corpus-expansion-75k`. Use separate data files and a separate Chroma collection, and a separate Neo4j database (Community edition has one user database, so a second DBMS or instance). Then re-measure the benchmarks.
5. Optional: evaluate retrieval with `CITES_ADDED`/`COUPLED_WITH` as an expansion edge type; deduplicate near-duplicate papers.

## Safe Commands

### Verify prerequisites and stores

```powershell
.\venv\Scripts\python.exe -c "from src.storage.neo4j_store import get_driver; d=get_driver(); d.verify_connectivity(); print('Neo4j OK'); d.close()"
.\venv\Scripts\python.exe -c "import chromadb; from src.utils.config import CHROMA_DIR, CHROMA_COLLECTION; print(chromadb.PersistentClient(path=CHROMA_DIR).get_collection(CHROMA_COLLECTION).count())"
ollama list
```

### Start the local demo

```powershell
ollama serve
.\venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Neo4j must already be running on `neo4j://127.0.0.1:7687`. Open `http://localhost:8501` after Streamlit starts.

### Frozen retrieval evaluation (complete)

```powershell
.\venv\Scripts\python.exe -m src.evaluation.refresh_retrieval evaluate --directory results\retrieval\expanded_20260913
.\venv\Scripts\python.exe -m src.evaluation.matched_rag --split test --output-dir results\reasoning\matched_rag_new
```

The judge protocol requires `OLLAMA_KEEP_ALIVE=2m`; the code default is `10m`, so set it for any grading process (`$env:OLLAMA_KEEP_ALIVE="2m"`). Do not run `prepare` in the existing `expanded_20260913` directory and do not edit its protocol, queries, candidates, ranks, or labels.

### Dev ablation and pool extension

```powershell
.\venv\Scripts\python.exe -m src.evaluation.variant_ablation --split dev --output-dir results\retrieval\variant_ablation_dev_new
.\venv\Scripts\python.exe -m src.evaluation.pool_extension --ablation-dir results\retrieval\variant_ablation_dev_new --split dev
```

Both refuse the test split without `--final-test`.

### Reproduce the offline reasoning package

```powershell
.\venv\Scripts\python.exe -m src.evaluation.packaged_reproduction --output-dir results\reasoning\package_reproduction_new
```

This requires no live database or LLM, but it reproduces AI-reference package results rather than human-ground-truth results.

### Reproduce headline numbers (no LLM calls)

```powershell
.\venv\Scripts\python.exe -m src.evaluation.reproduce
```

This recomputes 23 headline values (judge κ, human agreement and ratings, stratified and regraded NDCG, the LightRAG-style baseline, citation recommendation on our corpus and on DBLP/PubMed, time-sliced AP and P@10) from saved artifacts, and exits non-zero on drift. Two checks need the gitignored subset key in `evaluation/human_review/`.

### Verify code after changes

```powershell
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m compileall -q src app tests
.\venv\Scripts\python.exe -m pip check
```

## Data Safety Rules

- Never delete `data/s2_raw.json`, `data/s2_clean.json`, `data/s2_ner.json`, `data/chromadb/`, or `results/retrieval/expanded_20260913/` to recover from an interruption.
- Never run a second ingestion, NER, Chroma, or evaluation process while one is active. Each stage resumes from its durable state.
- `NEO4J_ALLOW_RESET=true` clears the configured graph. Use it only for an intentional rebuild of a dedicated project database.
- Keep `.env` and database credentials untracked. Do not expose tokens in logs, documentation, or commits.
- Do not treat results under `teammate2/` or `results/` as current live-store measurements unless their metadata confirms the same corpus snapshot.

## Documentation Map

| File | Purpose | Authority |
|---|---|---|
| `AGENT_HANDOFF.md` | Current scope, state, decisions, safe continuation path | Canonical |
| `PROJECT_STATUS.md` | Short current implementation/status report | Canonical summary |
| `README.md` | Public project overview and quick start | User-facing overview |
| `docs/RUN_FROM_SCRATCH.md` | Install, build every store, run the app, reproduce results | Setup authority |
| `docs/DEMO_QUERIES.md` | Queries that work well in each app mode | User guide |
| `docs/PAPER_DRAFT.md` | Paper draft | Paper authority |
| `SETUP.md` | Pointer to the run guide | — |
| `LOCAL_PROJECT_RUNBOOK.md` | Detailed local commands and interpretation | Operational reference; local-only |
| `docs/RESULTS.md` | All result tables with corpus, labels, and tests | Results authority |
| `docs/EVALUATION_PROTOCOL.md` | Metric definitions, selection rules, recorded selections | Protocol authority |
| `docs/LOCAL_EVALUATION.md` | Evaluation protocol and reproducibility commands | Evaluation authority |
| `Chinmay.md`, `devesh.md`, `VISHNU.md` | Contributor/historical work logs | Historical only |

## Instructions For a New Agent

1. Read this file, `PROJECT_STATUS.md`, and the relevant section of `docs/RUN_FROM_SCRATCH.md` before editing.
2. Run `git status --short --branch`; preserve user-owned uncommitted files and never reset/stash them without explicit need.
3. Confirm the active corpus and live-store counts before interpreting any evaluation result.
4. Keep production behavior, evaluation protocol, and historical artifacts separate. State which corpus, model, provider, and label source every result used.
5. Update this handoff, `PROJECT_STATUS.md`, README, setup/runbook references, and requirements when a material implementation or operational change is completed.
