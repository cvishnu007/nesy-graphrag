# NeSy-GraphRAG Project Status

Last updated: November 19, 2026. Detailed state and decisions: [AGENT_HANDOFF.md](AGENT_HANDOFF.md). Run guide: [docs/RUN_FROM_SCRATCH.md](docs/RUN_FROM_SCRATCH.md).

## Stage

**Research and evaluation complete; the paper draft is ready for venue formatting.** All planned experiments, baselines and human ratings are done. Remaining work is non-technical: choose a venue, add authors, and format the draft.

## System

| Component | Current choice |
|---|---|
| Corpus | 47,619 CS abstracts (Semantic Scholar, 2020–2025, 11 topics) |
| Retrieval | BGE-base + BM25 + graph expansion (`CITES`, `SIMILAR_TO`), reciprocal-rank fusion |
| Entities | SciBERT fine-tuned on SciERC (typed F1 0.64): 323,624 `SciEntity` nodes |
| Citations | Original `CITES` plus `CITES_ADDED` (133,527) and `COUPLED_WITH` (411,418) from Semantic Scholar and OpenAlex |
| Language model | Local `qwen2.5:7b-instruct` (Ollama) |
| App modes | Literature Review, Contradiction Detection (citation-grounded), Hypothesis Generation (method–task ranker), Related Work (co-citation, citation map) |
| Tests | 404 passing (no databases or models needed) |
| Reproduction | `python -m src.evaluation.reproduce`: 22/22 headline numbers |

## Results

Full tables in [docs/RESULTS.md](docs/RESULTS.md).

- **Topical retrieval:** BGE + BM25 + graph NDCG@10 0.644 vs 0.287 for the original system (44 queries). The graph adds a non-significant +0.023. Densifying citations adds −0.003 on test. A LightRAG-style GraphRAG baseline scores 0.396 (§2g).
- **Citation recommendation:** co-citation +28% Recall@20 on our corpus (§2f). On public DBLP: +0.069 Recall@20, F1@20 0.284 with no training vs 0.303 for the trained Citeomatic system. On PubMed it raises recall but lowers MRR, because text alone is already very precise there (§2h).
- **Time-sliced discovery:** graph signals nearly double average precision over text similarity (§4c–4d).
- **Contradictions:**
  - Round 1, NLI screening (two raters, κ 0.76): no agreed contradiction among 56 top pairs.
  - Round 2, citation-grounded index: 5 of 9 verdict-flagged pairs are a contradiction or tension (§5b-2).
- **Hypotheses:** graph-selected candidate papers are rated below vector neighbours in round 1 (two raters, κ 0.82–0.90) and round 2 (held-out queries) (§4b).
- **Answer grounding:** 41.8% vs 38.9% supported claims against plain vector RAG, not significant (§5c-2).
- **Judge:** a local 7B judge reaches held-out κ 0.75 against human grades; two humans agree at κ 0.91 (§2b).

## Branches and tags

| Name | Contents |
|---|---|
| `master` | Current state: everything below merged |
| `research-grade-evaluation` | Paper state before the two new baselines |
| `baselines-graphrag-citation` | Adds the LightRAG-style baseline and the DBLP/PubMed benchmarks |
| `corpus-expansion-75k` | Placeholder for expanding the corpus to 75k papers (not started) |
| tags `paper-freeze-2026-11-19`, `app-freeze-2026-11-19` | Frozen states before the new baselines |

## Open items

- **Decisions:** the venue (and formatting to its template), and the authors.
- **Optional:**
  - a second rater for round 2;
  - more labelled contradiction pairs (the 56% rests on 9 flagged pairs);
  - human ratings of the hypothesis ranker;
  - expanding the corpus to 75k papers.

## Evidence boundaries

- Relevance labels are LLM-judge labels validated against humans, not human ground truth.
- Round 2 human ratings come from one rater.
- The GraphRAG baseline re-implements LightRAG's retrieval over our entity graph; the original LLM-built index was infeasible on our hardware.
- The DBLP/PubMed comparison is with 2018 systems, and BGE's training data likely overlaps with those papers.
