# NeSy-GraphRAG Project Status

Last updated: August 29, 2026

This is the single status file for the project. PDF ingestion is out of scope for the current phase.

## Current State

The full S2 pipeline has been rerun from scratch and the system runs end to end.

Fresh run completed:

- Semantic Scholar ingestion:
  - fetched 10,000 papers
  - kept 8,850 papers after cleaning
  - wrote `data/s2_raw.json`
  - wrote `data/s2_clean.json`
- NER:
  - processed 8,850 papers
  - extracted entities for 8,850 papers
  - wrote `data/s2_ner.json`
- ChromaDB:
  - reset stale `s2_papers` collection
  - rebuilt `s2_papers`
  - final vector count: 8,850
- Neo4j:
  - cleared old graph data
  - reloaded from fresh `data/s2_ner.json`
  - final graph counts:
    - Papers: 8,850
    - Authors: 27,655
    - Concepts: 43,581
    - `AUTHORED_BY`: 37,180
    - `RELATED_TO`: 88,268
    - real `CITES`: 7,203
    - simulated `CITES`: 0
- End-to-end pipeline:
  - review mode completed
  - contradiction mode completed
  - hypothesis mode completed
  - baseline comparison completed on 5 queries
- Compute environment:
  - RTX 3050 Laptop GPU detected with 4 GB VRAM
  - automatic GPU/CPU selection is implemented
  - PyTorch `2.12.1+cu126` and torchvision `0.27.1+cu126` are installed
  - `torch.cuda.is_available()` returns `True`
  - live SPECTER retrieval automatically selected CUDA successfully
  - measured model/query memory: about 429 MB allocated and 474 MB reserved VRAM

## Main Finding

The retrieval ranking defect has been fixed. Chroma now returns real cosine similarity, graph candidates receive normalized connectivity scores, and the final ranking uses reciprocal-rank fusion.

Live result for `graph neural networks for node classification`:

- before: 10 neural, 0 symbolic/both, NBR `0.0`
- after: 5 neural, 4 symbolic, 1 both, NBR `0.5`
- Neo4j and the existing 8,850-vector Chroma collection were used for this check

The corrected five-query baseline comparison completed successfully. Average NeSy deltas versus vector-only retrieval were:

- TS delta: `0.0`
- NBR delta: `+0.56`
- ATD delta: `-0.12`
- RDI delta: `+0.10`

Per-query NeSy NBR was `0.5`, `0.5`, `0.5`, `0.7`, and `0.6`; every baseline NBR was `0.0`. Trustworthiness remained `1.0` for both paths. The graph now measurably affects retrieval, but reduced temporal diversity on three queries and low absolute RDI require further work.

This proves that graph expansion participates in ranking. It does **not** yet prove that the extra papers are more relevant. The five-query run has no human relevance labels, so NBR cannot substitute for Precision@K, Recall@K, MRR, or NDCG.

## Alignment With The Phase 1 Goal

| Phase 1 expectation | Current position | Remaining gap |
|---|---|---|
| Automated literature review | Working end to end | Evaluate review completeness and factual quality |
| Graph-grounded attribution | Review claims cite deterministic sentences from Neo4j-verified paper abstracts | Benchmark semantic support; add section provenance only if full text enters scope |
| Cross-paper contradiction reasoning | Candidate ranking and structured verdicts work | Benchmark against labeled scientific NLI/contradiction data |
| Hypothesis generation | Evidence-ranked generation and feasibility fields work | Expert or benchmark validation of novelty and feasibility |
| End-to-end evaluation | Custom metrics and vector baseline exist | Add relevance judgments, standard IR metrics, repeated runs, and human evaluation |
| Large scientific corpus | 8,850 mostly CS records are indexed | Scale only after quality and metric logic are validated |
| Model training for reasoning | Pretrained SPECTER plus hosted Llama are used | Training/fine-tuning has not been implemented |

PDF ingestion remains intentionally deferred. Claim provenance is implemented over the abstracts already stored, without expanding ingestion scope.

## What Is Implemented

### Ingestion

- ArXiv ingestion in `src/ingestion/arxiv_fetcher.py`
- Semantic Scholar ingestion in `src/ingestion/semantic_scholar_fetcher.py`
- Source switching through `DATA_SOURCE`
- Fresh S2 local dataset with references

### Entity Extraction

- spaCy-based entity and noun-chunk extraction in `src/ingestion/ner_extractor.py`
- Multiprocessing support through `SPACY_N_PROCESS`
- Batch size control through `SPACY_BATCH_SIZE`
- Automatic spaCy GPU preference with parallel CPU fallback

### Vector Store

- ChromaDB persistent indexing in `src/storage/chroma_store.py`
- SPECTER embedding model loading
- Resume support for partially indexed collections
- Automatic SPECTER device selection in CUDA, MPS, CPU order
- Conservative embedding micro-batches for limited GPU memory
- PyTorch CPU thread controls:
  - `TORCH_NUM_THREADS`
  - `TORCH_INTEROP_THREADS`
  - `OMP_NUM_THREADS`
  - `MKL_NUM_THREADS`
- Offline cached Hugging Face model loading for stable local runs

### Knowledge Graph

- Neo4j loader in `src/storage/neo4j_store.py`
- Nodes:
  - `Paper`
  - `Author`
  - `Concept`
- Edges:
  - `AUTHORED_BY`
  - `RELATED_TO`
  - real `CITES`
  - simulated fallback `CITES`
- Current graph uses real Semantic Scholar citations only

### Retrieval

- Neural retrieval through ChromaDB
- Symbolic expansion through Neo4j `CITES` traversal
- Real Chroma cosine similarities instead of fixed scores
- Normalized graph connectivity scores
- Weighted reciprocal-rank fusion
- Source labels:
  - `neural`
  - `symbolic`
  - `both`
- Vector-only baseline retrieval
- One-command retrieval diagnostics with:
  - neural and graph ranks
  - neural similarity and graph connections
  - citation degree
  - final source distribution and score
  - kept/dropped cutoff decision

### Citation Validation

- `validate_citations()` verifies retrieved paper IDs against Neo4j
- LLM review context only includes verified papers
- Latest run verified `10/10` retrieved citations in review mode

### Claim Provenance

- verified abstracts are split into deterministic sentence passages
- passage IDs are stable hashes of paper ID plus sentence position
- review prompts contain exact passage IDs and require a citation for every claim
- only claims whose complete citation set exists in the supplied passage index are displayed
- missing, malformed, mixed-validity, and fabricated passage citations are blocked
- generation is deterministic and one bounded format-repair attempt is allowed only when the first response yields zero valid claims
- rejected claims, parse errors, raw output, and passage metadata are retained for audit
- TS now uses passage-citation integrity and grounded-claim coverage
- Streamlit exposes the supporting sentence and paper for every accepted claim
- latest live smoke result: 23 passages, 5/5 grounded claims, 9/9 valid citations, 0 unsupported claims, 0 parse errors, and no repair needed

### LLM Modes

- Literature review mode
- Contradiction detection mode
- Hypothesis generation mode
- Shared prompt templates
- Groq retry and fallback wrapper

### Contradiction Detection

- Cross-year graph candidate generation
- Candidate ranking by normalized concept Jaccard and year gap
- Configurable minimum shared concepts and overlap threshold
- Exact structured verdict parser for:
  - `CONTRADICTION`
  - `AGREEMENT`
  - `DIFFERENT SCOPE`
- Deterministic LLM temperature for contradiction checks
- Configurable confidence gate before a contradiction contributes to RDI
- Shared verdict interpretation across pipeline, metrics, and Streamlit
- Live check returned two valid `DIFFERENT SCOPE` verdicts with confidence `0.92` and `0.96`

### Hypothesis Validation

- Structural holes must be supported by at least two neural seed papers
- Candidate score combines normalized concept overlap and query-paper support
- Shared concept names and supporting paper IDs are retained as evidence
- Deterministic structured generation requires:
  - feasibility
  - supporting evidence
  - missing evidence
- Only valid `HIGH` or `MEDIUM` hypotheses enter normal results and HNS
- Rejected or malformed generations remain available under `rejected_hypotheses`
- Streamlit displays evidence score, supporting-paper count, feasibility, and missing evidence
- Live check returned a valid accepted `MEDIUM`-feasibility hypothesis

### Metrics

- `TS`: Trustworthiness Score
- `NBR`: NeSy Boost Ratio
- `ATD`: Answer Temporal Diversity
- `RDI`: Reasoning Depth Index
- `HNS`: Hypothesis Novelty Score
- Centralized contradiction verdict parsing

### UI

- Streamlit app exists
- Supports:
  - literature review
  - contradiction detection
  - hypothesis generation
  - metrics display
  - retrieval source badges
  - basic graph stats

## Top Priority Implementation Work

### 1. Build A Judged Retrieval Evaluation

The corrected fusion produces strong graph contribution, but relevance has not been measured and ATD averaged `-0.12` versus baseline.

Next implementation:

- create a fixed query set with manually judged relevant papers
- report Precision@K, Recall@K, MRR, and NDCG for vector-only and hybrid retrieval
- sweep fusion weights and graph expansion depth on the same judgments
- report source mix, NBR, ATD, and latency as diagnostics rather than relevance substitutes
- separate retrieval comparisons from hosted-LLM variation

### 2. Correct Metric Semantics

- benchmark or replace prototype TS with a semantic claim/evidence support measure
- treat NBR strictly as a graph-contribution diagnostic
- replace or justify the custom RDI formula
- correct HNS: the current `1 / path_length` implementation rewards shorter paths while the documentation describes longer paths as more novel
- document metric ranges, edge cases, and interpretation with tests

### 3. Benchmark Reasoning Modes

- evaluate contradiction verdicts on labeled scientific claim pairs or a small expert-labeled set
- measure precision, recall, F1, calibration, and malformed-output rate
- evaluate hypotheses separately for evidence support, novelty, feasibility, and usefulness
- retain human review as the final gate for scientific claims

## Automated Test Coverage

- 33 pytest cases pass on the default Windows console
- pytest discovery and strict marker configuration are defined in `pytest.ini`
- shared test fixtures are defined in `tests/conftest.py`
- retrieval fusion and diagnostics
- structured contradiction scoring, parsing, and confidence gating
- fabricated citation IDs are blocked
- empty metric inputs return defined zero values, including TS `0.0`
- missing Neo4j credentials fail before driver creation
- connectivity errors close the driver and report the URI without exposing credentials
- hypothesis evidence scoring, structured feasibility parsing, and rejection auditing
- stable passage IDs, lowercase sentence splitting, strict claim parsing, fabricated passage blocking, provenance-aware TS, and review-level filtering

The Neo4j unit tests use isolated in-memory driver doubles only for deterministic guard/failure behavior. Live retrieval and contradiction checks use the real local Neo4j database.

## Secondary Work

### Scientific NLI Evaluation

The structured contradiction layer is implemented, but verdict semantics still depend on the hosted LLM. Compare it against a scientific NLI or claim-level classifier before making research-grade contradiction claims.

### HNS Refinement

HNS currently measures graph path structure only. Incorporate accepted feasibility/evidence information or report it as a separate quality metric before treating HNS as a complete hypothesis-quality score.

### Entity Extraction Quality

Current spaCy extraction works but is generic. It extracts many noun chunks that are not precise scientific concepts.

Future improvement:

- try scispaCy or SciBERT-style concept extraction
- compare graph quality before and after
- measure effect on hypothesis quality and fallback citation edges

### Graph Coverage And Scale

- diagnose why only 2,989 of 8,850 papers participate in citation edges
- improve citation/concept coverage before increasing corpus size
- compare one-hop and bounded multi-hop expansion under the judged benchmark
- scale the corpus only after retrieval quality is stable; sparsity will not automatically fix ranking logic

### Engineering Reliability

- add live integration tests for Neo4j and Chroma behind explicit test markers
- pin or constrain critical dependency versions for reproducible installs
- add CI for unit tests and formatting
- export evaluation runs and parameters in a comparison-friendly format

### UI Improvements

Useful but not urgent:

- graph visualization
- source distribution panel
- retrieved-vs-expanded comparison
- clearer metric explanations

## Known Weak Points

- Current real citation graph is sparse:
  - only 2,989 of 8,850 papers participate in at least one `CITES` edge
  - 1,581 papers have outgoing real `CITES` edges
- Equal neural/graph fusion weights may need tuning after the five-query evaluation.
- Graph-expanded papers need qualitative relevance checks, not only a higher NBR.
- Claim provenance validates citation IDs and traceability, but does not itself prove semantic entailment.
- The current contradiction module is heuristic-first, not a proper contradiction model.
- The current dataset is abstracts/metadata at prototype scale, not the report's proposed large full-text corpus.
- Current concept extraction is generic spaCy noun-phrase extraction and introduces noisy graph nodes.
- Hypothesis feasibility is an LLM judgment, not experimental or expert validation.
- HNS currently has a direction mismatch between its formula and documented interpretation.

## Current Readiness

- Working demo readiness: about 95%
- Capstone prototype implementation readiness: about 90%
- Defensible capstone evaluation readiness: about 62%
- Research-grade readiness: about 35%

The core implementation is now sealed as a working prototype: all three user-facing modes execute, graph retrieval contributes real results, GPU inference works, claim-level abstract provenance blocks fabricated evidence references, and core contracts have tests. Remaining work is experimental validation: concrete benchmarks, standard metrics, classical and modern baselines, controlled ablations, measured scaling, and evaluation-focused UI/reporting.
