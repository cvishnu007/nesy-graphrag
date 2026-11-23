# NeSy-GraphRAG Project Status

Last updated: November 20, 2026. Detailed state and decisions: [AGENT_HANDOFF.md](AGENT_HANDOFF.md). Run guide: [docs/RUN_FROM_SCRATCH.md](docs/RUN_FROM_SCRATCH.md).

## Stage

**Draft; figures done, 46 of 48 references checked.** The planned experiments, baselines and human ratings are done and an internal audit's blockers are addressed. Open: regrading the 44-query pool with the 27B judge, the leakage reruns and per-query outputs (need the local stores), two references, then venue, authors and formatting.

## System

| Component | Current choice |
|---|---|
| Corpus | 47,619 CS abstracts (Semantic Scholar, 2020–2025, 11 topics) |
| Retrieval | BGE-base + BM25 + graph expansion (`CITES`, `SIMILAR_TO`), reciprocal-rank fusion |
| Entities | SciBERT fine-tuned on SciERC (typed F1 0.64): 323,624 `SciEntity` nodes |
| Citations | Original `CITES` plus `CITES_ADDED` (133,527) and `COUPLED_WITH` (411,418) from Semantic Scholar and OpenAlex |
| Language model | Local `qwen2.5:7b-instruct` (Ollama) |
| App modes | Literature Review, Contradiction Detection (citation-grounded), Hypothesis Generation (method–task ranker), Related Work (co-citation, citation map); a query gate rejects generic, non-research input before any mode runs |
| Tests | 423 passing (no databases or models needed) |
| Reproduction | `python -m src.evaluation.reproduce`: 35/35 headline numbers, 26 recomputed from raw files, 9 read from saved summaries |

## Results

Full tables in [docs/RESULTS.md](docs/RESULTS.md).

- **Topical retrieval:** BM25 0.626, BGE + BM25 0.621 and with the graph 0.644 NDCG@10 (44 queries); the graph adds a non-significant +0.023, and the gain over BGE alone is not significant after Holm (p = 0.0104). A small cross-encoder reranker on the top 20 gives 0.651, about the same (§2i). Densifying citations adds −0.003 on test. A LightRAG-style GraphRAG baseline scores 0.396 (§2g).
- **Citation recommendation:** co-citation +28% Recall@20 on our corpus (§2f). On public DBLP and PubMed it raises Recall@20 on top of BGE, SPECTER2 and SciNCL (citation-trained encoders). With SPECTER2 (itself citation-pretrained) on DBLP it comes close to the trained Citeomatic system (F1@20 0.300 vs 0.303; MRR 0.708 vs 0.689). On PubMed it lowers MRR, because text alone is already very precise there (§2h).
- **Time-sliced discovery:** graph signals nearly double average precision over text similarity (§4c–4d).
- **Contradictions:**
  - Round 1, NLI screening (two raters, κ 0.76): no agreed contradiction among 56 top pairs.
  - Round 2, citation-grounded index: 5 of 9 verdict-flagged pairs are a contradiction or tension (§5b-2).
- **Hypotheses:** graph-selected candidate papers are rated below vector neighbours in round 1 (two raters, κ 0.82–0.90) and round 2 (held-out queries) (§4b).
- **Answer grounding:** 41.8% vs 38.9% supported claims against plain vector RAG, not significant (§5c-2).
- **Judge:** a local 7B judge reaches held-out κ 0.75 on GNN pairs but 0.57 on the other ten topics; a 27B judge reaches 0.88 on both; two humans agree at κ 0.91 (§2b).

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
- **27B regrade of the 44-query pool:** `python -m src.evaluation.regrade_stratified grade --max-pairs 190` once a day on the machine with `data/` and the Groq key (about 12 days; topic round-robin, so a partial run covers every topic), then `score`.
- **Reruns on the machine with the stores** (each writes a new folder and per-query metrics, so `reproduce.py` then recomputes all 35 checks):
  - `python -m src.evaluation.citation_recommendation` then `--final-test` (frozen setting, now with per-query output), and both again with `--exclude-near-duplicates`;
  - `python -m src.evaluation.citeomatic_benchmark --dataset {dblp,pubmed} --split test [--encoder specter2_base]`;
  - `python -m src.evaluation.temporal_hypothesis`, and again with `--types-from-past --output-name summary_types_from_past.json`.
- **Commit the round-2 hypothesis key** (`evaluation/human_review/hypothesis_round2_key_DO_NOT_SHARE_WITH_ANNOTATORS.json`; rating is finished). The other arm keys are rebuilt by `rebuild_arm_keys.py`.
- **Demo:** set `NEO4J_ALLOW_RESET=false` in the local `.env`, rehearse on the demo laptop, record a backup video of the four modes (`docs/DEMO_QUERIES.md`).
- **Contribution split:** each member confirms or corrects their row below.
- **Optional:**
  - a second rater for round 2;
  - more labelled contradiction pairs (the 56% rests on 9 flagged pairs);
  - human ratings of the hypothesis ranker;
  - expanding the corpus to 75k papers.

## Contribution split (draft from git history, to confirm)

Commit counts miss design, writing and rating work, so this is only a starting point. Rows are by git account; replace with names and one line each.

| Git account | Commits | Period | Areas in the history |
|---|---:|---|---|
| cvishnu007 / Vishnu Chiyedu | 82 | Apr–Nov 2026 | Pipelines, evaluation protocol and benchmarks, judge validation, human-review kits, app, paper draft |
| chinmaydwivedi | 8 | Apr–Sep 2026 | Reasoning evaluation (`results/reasoning`), pipeline and evaluation modules, audit of the November state |
| Devesh-Reddy99 | 16 | Aug 2026 | Retrieval baselines and tuned hybrid retrieval, evaluation modules |
| geekygruesome | 5 | Sep 2026 | `teammate2/` evaluation package: annotations, tooling, tests |

## Evidence boundaries

- Relevance labels are LLM-judge labels, not human ground truth; the 7B judge was validated on GNN-topic pairs. On 100 pairs from the other ten topics the judge reaches κ 0.57, below the 0.6 bar; it is strict, equally so with and without the graph, more so for the original system (RESULTS §2b).
- Raters are anonymous. Rater A (every task) is the system's developer; Rater B is a project member who did not build it; Rater C graded the judge-transfer check (`evaluation/human_review/README.md`, Rater record).
- The co-citation expansion is prior work (Strohman et al., 2007; Citeomatic's NNSelect); the paper claims only that it still adds to modern encoders.
- Round 2 human ratings come from one rater.
- The GraphRAG baseline re-implements LightRAG's retrieval over our entity graph; the original LLM-built index was infeasible on our hardware.
- The DBLP/PubMed published systems are from 2018; the modern baselines are encoders (BGE, SPECTER2, SciNCL), not trained citation recommenders, and their training data likely overlaps with those papers.
