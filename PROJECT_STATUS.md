# NeSy-GraphRAG Project Status

Last updated: August 28, 2026

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

The previous five-query baseline comparison was run before this fix and is now obsolete. Its deltas were:

- TS delta: `0.0`
- NBR delta: `0.0`
- ATD delta: `0.0`
- RDI delta: `0.0`

The next evaluation run must determine whether the improved source mix also improves answer-level metrics and qualitative review quality.

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

### Citation Validation

- `validate_citations()` verifies retrieved paper IDs against Neo4j
- LLM review context only includes verified papers
- Latest run verified `10/10` retrieved citations in review mode

### LLM Modes

- Literature review mode
- Contradiction detection mode
- Hypothesis generation mode
- Shared prompt templates
- Groq retry and fallback wrapper

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

### 1. Re-run Baseline Comparison After Ranking Fix

Run the existing five-query harness and preserve the new results. Record TS, NBR, ATD, RDI, source distribution, and a short qualitative comparison for each query.

Success criteria:

- NeSy and baseline retrieval sets are no longer identical
- graph contribution remains measurable across multiple queries
- answer quality does not regress from admitting graph-expanded papers

### 2. Add Retrieval Diagnostics

Before changing deeper logic, make the pipeline easier to inspect.

Add debug output or a small diagnostic function showing:

- neural top-k IDs and titles
- symbolic expanded IDs and titles
- citation degree for retrieved papers
- final merged ranking
- source distribution
- why symbolic candidates were kept or dropped

Success criteria:

- one command can explain why a query produced NBR `0.0`
- ranking changes can be evaluated without guessing

### 3. Improve Contradiction Detection

Current behavior:

- finds papers that share concepts and come from different years
- asks the LLM if they contradict

This is useful for a demo but weak for evaluation.

Next implementation:

- keep current graph-based candidate generation
- add a stricter NLI/contradiction classifier or structured scoring layer
- keep the existing output shape so UI and metrics do not break

Success criteria:

- fewer irrelevant contradiction pairs
- reproducible contradiction labels
- better RDI credibility

### 4. Expand Automated Tests

Add lightweight tests before deeper refactors.

Already covered:

- graph-only results can enter top-k
- overlap is labelled `both` and boosted
- fusion does not mutate its inputs

Still needed:

- fabricated paper ID is blocked by `validate_citations()`
- contradiction verdict parser does not count negated mentions as contradictions
- metric functions handle empty results
- Chroma/Neo4j connection failures fail clearly

Success criteria:

- core claims can be defended without relying only on manual runs

## Secondary Work

### Entity Extraction Quality

Current spaCy extraction works but is generic. It extracts many noun chunks that are not precise scientific concepts.

Future improvement:

- try scispaCy or SciBERT-style concept extraction
- compare graph quality before and after
- measure effect on hypothesis quality and fallback citation edges

### Hypothesis Validation

Current hypothesis mode finds structural holes and asks the LLM to generate hypotheses.

Future improvement:

- add feasibility checks
- score novelty more transparently
- filter weak hypotheses
- include supporting and missing evidence

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
- TS is strong, but currently measures paper-level validation, not claim-level provenance.
- The current contradiction module is heuristic-first, not a proper contradiction model.

## Current Readiness

- Working demo readiness: about 85%
- Capstone evaluation readiness: about 72%
- Research-grade readiness: about 45%

The next milestone is validating the corrected hybrid retrieval across the fixed evaluation query set, then tuning it based on evidence.
