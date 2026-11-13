# NeSy-GraphRAG Continuation Handoff

Last reconciled: November 14, 2026. Read this file before changing code, data, configuration, or documentation. It is the canonical handoff for a new chat or coding agent. `PROJECT_STATUS.md` is the concise status view; `SETUP.md` is the from-scratch setup guide; `LOCAL_PROJECT_RUNBOOK.md` is a detailed local operations reference.

## Mission and Scope

The Phase 1 report defines the project as a trustworthy research assistant that:

- synthesizes scientific literature with traceable evidence;
- reasons across papers to surface possible contradictions;
- generates evidence-based research hypotheses; and
- evaluates reliability rather than treating fluent LLM output as scientific truth.

The Phase 1 report (August-December 2025) listed noisy-PDF extraction and model training as Phase 2 objectives. The team later deferred both, so they are not part of the current scope.

The current implementation is an abstract-and-metadata prototype. It is deliberately **not** a PDF/full-text ingestion system, a trained/fine-tuned model, a fully autonomous research agent, or a source of validated scientific conclusions. Do not add PDF ingestion unless the user explicitly reopens that scope.

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
| Retrieval embedding | BGE-base-en-v1.5 (`s2_papers_bge`, 47,619 vectors) | Beats SPECTER, SPECTER2 base, SciNCL, and OpenScholar on SciDocs and LitSearch (all p < 0.001); project test NDCG@10 0.619 vs 0.449 for the previous production (p = 0.004) |
| Retrieval fusion | BGE + BM25 + graph (CITES and SIMILAR_TO, relative gate) | Dev-selected, test-confirmed once |
| Entity extraction | SciBERT fine-tuned on SciERC (`models/scibert-scierc-ner`, gitignored) | SciERC test typed F1 0.639 vs GLiNER 0.240 and community checkpoints ≤ 0.294; spaCy span F1 0.270 |
| Concept graph | `(:Paper)-[:MENTIONS]->(:SciEntity {name, type, df, generic})`: 323,624 nodes, 696,390 edges | Shared concepts become task/method terms instead of discourse phrases |
| Contradiction candidates | NLI over authors' finding sentences (`src/pipeline/contradiction_nli.py`) | Concept overlap yields few or noise-driven pairs; NLI returns sentence evidence |

Rebuild order for the scientific concept graph: `python -m src.ingestion.train_scierc_ner` (needs `data/external/zj88zj__SCIERC/{train,dev,test}.json`), then `python -m src.ingestion.scientific_ner extract --model models/scibert-scierc-ner`, then `python -m src.storage.graph_enrichment scientific-entities --write`. The historical spaCy `Concept` graph remains in Neo4j; set `CONCEPT_SOURCE=spacy` to use it. Remove the new graph with `MATCH (e:SciEntity) CALL (e) { DETACH DELETE e } IN TRANSACTIONS OF 5000 ROWS` (run in an auto-commit session).

End-to-end smoke test (`results/local/smoke_test_20261114.log`): review 6/6 claims grounded with 12/12 valid citations from neural, lexical, symbolic, and overlapping sources; contradiction 5/5 valid structured verdicts; hypothesis 5 accepted. The local 3B model reports confidence 1.0 for every verdict, so verdict confidence is uncalibrated.

**Environment notes.** `gliner`, `sentencepiece`, and `protobuf==7.36.1` were added (protobuf must be at least the gencode version sentencepiece was built with). The `adapters` package was not installed because it would downgrade `transformers`. The local environment runs spaCy 3.7.5 although `requirements.txt` pins 3.8.16; this predates the November work. Hugging Face's Xet client hung on some downloads; `HF_HUB_DISABLE_XET=1` works around it. Unit tests pin all selectable options in `tests/conftest.py`, so they do not depend on `.env`.

## Important Limitations

- Citation expansion is sparse. More papers broaden semantic retrieval but do not automatically create useful graph contribution.
- spaCy noun chunks are noisy concept proxies; concept density is not concept quality.
- Existing retrieval judgments are local Ollama AI-reference labels, not human relevance ground truth.
- The old 20-query set concerns graph-neural-network topics and is not representative of the broad CSE corpus.
- The frozen in-domain contradiction package has inadequate positive-class coverage; do not make strong contradiction-quality claims from it.
- Claim provenance proves that a cited passage exists and is structurally valid. It does not by itself prove full semantic entailment.
- Hypotheses, contradictions, and reviews are model outputs requiring human/expert review.

## Immediate Next Work

Items 1-3 of the earlier plan (start Ollama, finish grading, score with significance) were completed on November 13, 2026. Continue in this order unless the user changes priorities; the full publication roadmap is in [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md):

1. Have the team review `retrieval_queries_v2_stratified_draft.json`, then pool and grade it in a fresh directory.
2. Grade a 200-pair human sample of the frozen pool to measure AI-judge agreement.
3. Deduplicate near-duplicate papers.
4. Run matched-budget vector RAG versus GraphRAG with the frozen protocol. It runs live on the current 47,619-paper stores; record that corpus in the write-up.
5. Obtain independent human review of relevance judgments and generated answers before reporting research-quality conclusions.
6. Build a better-covered in-domain contradiction benchmark and expert review protocol for hypotheses.
7. Compare scientific NER, embedding, and LLM alternatives using frozen development/test protocols, one component at a time.
8. Scale in controlled 10K/50K/100K experiments only after retrieval-quality evidence is stable.

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
| `SETUP.md` | Clone-to-running-environment instructions | Setup authority |
| `LOCAL_PROJECT_RUNBOOK.md` | Detailed local commands and interpretation | Operational reference; local-only |
| `docs/RESULTS.md` | All result tables with corpus, labels, and tests | Results authority |
| `docs/EVALUATION_PROTOCOL.md` | Metric definitions, selection rules, recorded selections | Protocol authority |
| `docs/LOCAL_EVALUATION.md` | Evaluation protocol and reproducibility commands | Evaluation authority |
| `Chinmay.md`, `devesh.md`, `VISHNU.md` | Contributor/historical work logs | Historical only |

## Instructions For a New Agent

1. Read this file, `PROJECT_STATUS.md`, and the relevant subsection of `SETUP.md` before editing.
2. Run `git status --short --branch`; preserve user-owned uncommitted files and never reset/stash them without explicit need.
3. Confirm the active corpus and live-store counts before interpreting any evaluation result.
4. Keep production behavior, evaluation protocol, and historical artifacts separate. State which corpus, model, provider, and label source every result used.
5. Update this handoff, `PROJECT_STATUS.md`, README, setup/runbook references, and requirements when a material implementation or operational change is completed.
