# Human Review Tasks

**Status (2026-11-19):** all planned ratings are done and scored (RESULTS §2b, §4b, §5b-2). Excluded template/AI sheets are in `excluded/` with the reasons. Open: the judge transfer check (task 9) and the rater record below. Optional: a second rater for round 2.

These human labels are what make the evaluation publishable: they validate the automated judges and measure the two components that have no public benchmark (hypotheses, and contradictions in our CS corpus).

**Rules for every sheet**
- Work **independently**. Do not discuss items until every rater has finished.
- Judge only from the text in the sheet. Do not search the web.
- Do **not** open any `*_key_DO_NOT_SHARE_WITH_ANNOTATORS.json` file. The keys say which method produced each row; seeing them biases the review.
- Copy the blank sheet to `<name>_sheet_<initials>.csv`, fill only the rating columns, and save as CSV (UTF-8). If Excel asks, keep the CSV format.
- When unsure between two values, choose the lower one and write a short note.

## Status

| # | Task | Who | Time | Status |
|---|---|---|---|---|
| 1 | Relevance grades, 200 pairs | CV | 2–3 h | ✅ Done (`relevance_sheet_cv.csv`). Selected the AI judge: κ 0.88 held out |
| 2 | Review the 44-query stratified benchmark | CV | 1 h | ✅ Done (v0.5, `evaluation/benchmarks/`) |
| 3 | Second relevance annotator, same 200 pairs | Teammate (not CV) | 2–3 h | ⚠️ Returned 2026-11-19, but started from an AI/template pass (36 of 200 grades revised by hand); not usable as independent agreement. Replaced by task 3b |
| 3b | **Independent** second relevance grader, 50-pair subset | Teammate (not CV) | ~40 min | ✅ Done 2026-11-19 (`relevance_subset50_sheet_rater3.csv`): human–human κ 0.91 |
| 4 | Hypothesis ratings, 60 items | 2 people | ~1.5 h each | ✅ Done (`hypothesis_sheet_rated_cv.csv`, `hypothesis_sheet_rated_rater3.csv`); κ 0.82–0.90 |
| 5 | Contradiction labels, 68 pairs | 2 people | ~1 h each | ✅ Done (`contradiction_sheet_labeled_cv.csv`, `contradiction_sheet_labeled_rater3.csv`); κ 0.76 |
| 6 | Free OpenAlex API key | Project owner | 5 min | ⏳ Unblocks citation densification for papers whose references Semantic Scholar hides |

| 7 | **Round 2** hypothesis ratings, 42 items (14 new test queries) | 2 people | ~1 h each | ✅ Rater 1 done (`hypothesis_round2_sheet_rated_cv.csv`); second rater optional |
| 8 | **Round 2** contradiction labels, 40 pairs | 2 people | ~45 min each | ✅ Rater 1 done (`contradiction_round2_sheet_labeled_cv.csv`); second rater optional |

| 9 | **Judge transfer check**: relevance grades, 100 pairs from the 10 non-GNN topics | Anyone except CV preferred; CV acceptable | ~1.5 h | ⏳ Open (`judge_transfer_sheet_blank.csv`). Tests whether the 7B judge, validated on GNN pairs, holds on the other topics of the 44-query benchmark |

Round 2 tests the methods revised after round 1, on queries not used for the revision. Rate it like round 1; the sheets do not say which method produced a row.

For tasks 4, 5, 7 and 8, two raters are needed for agreement. If only one person is available, the results are still useful but are reported as single-rater.

---

## Task 3: Second relevance annotator

Sheet: `relevance_sheet_blank.csv`. For each row, read the query, then the paper's title and abstract, and enter `grade_0_1_2`:

| Grade | Meaning | Example for "graph neural networks for fraud detection" |
|---|---|---|
| **2** | The paper's main contribution addresses the whole query | A GNN model for detecting financial fraud |
| **1** | Addresses part of the query, or the topic is secondary | A GNN for general anomaly detection; a non-GNN fraud detector |
| **0** | Does not address the query's core subject | A GNN for molecules; a survey of blockchain |

Shared words are not enough: "network traffic" is not "road traffic".

### Task 3b: 50-pair subset (independent grader)

Sheet: `relevance_subset50_sheet_blank.csv`. Same 0/1/2 scale as above. For the result to count as independent agreement:

- Grade every row yourself, from the query, title and abstract only.
- Do not use ChatGPT, Claude, Copilot or any other AI tool, and do not run a script to fill grades or notes.
- Do not open `relevance_sheet_cv.csv`, other graded sheets, or anything named `*DO_NOT_SHARE*`.
- Notes are optional; a few words for uncertain rows is enough.

Return the file renamed `relevance_subset50_sheet_<initials>.csv`.

### Task 9: Judge transfer check (100 pairs, 10 topics)

Sheet: `judge_transfer_sheet_blank.csv`. Same 0/1/2 scale and the same independence rules as task 3b. The queries cover cybersecurity, databases, NLP, vision, networks, cloud, operating systems, software engineering, data science and general ML; judge each from the query, title and abstract only. Return it renamed `judge_transfer_sheet_<initials>.csv` and score it with:

```powershell
.\venv\Scripts\python.exe evaluation\human_review\build_judge_transfer_sheet.py score --sheet evaluation\human_review\judge_transfer_sheet_<initials>.csv
```

The predeclared bar is the same as for the first validation: quadratic-weighted κ ≥ 0.6 between the human and the stored 7B judge labels. Report the result whatever it is; below 0.6, the 44-query numbers must be described as resting on an unvalidated judge for those topics.

## Task 4: Hypothesis ratings

Sheet: `hypothesis_sheet_blank.csv` (60 rows; 30 queries, 2 candidates each, shuffled).

Each row is a possible research direction: a researcher who works on the **query** and knows the **query papers** is shown a **candidate paper**. The question is whether combining the candidate's ideas with the query topic suggests a worthwhile new research direction. Rate 1–5:

| Column | 1 | 3 | 5 |
|---|---|---|---|
| `relevance_1_5`: could the candidate's ideas plausibly be applied to the query topic? | No meaningful connection | Some connection, needs a stretch | Clear, direct way to apply it |
| `novelty_1_5`: is the combination non-obvious? | Same topic; already the standard approach | Somewhat new | Surprising, and nobody in the query papers seems to do it |
| `promise_1_5`: would you want to read a paper that tries it? | No | Maybe | Yes, likely worthwhile |

A candidate can be relevant but not novel (a near-duplicate of the query papers), or novel but irrelevant. Rate the three columns separately.

## Task 5: Contradiction labels

Sheet: `contradiction_sheet_blank.csv` (68 rows). Each row has two findings from two different papers. Enter `label_0_1_2`:

| Label | Meaning | Example |
|---|---|---|
| **2** | Real contradiction: same question, incompatible findings | "Method X improves accuracy on dataset D" vs "Method X does not improve accuracy on D" |
| **1** | Tension: findings point in different directions, but different settings, datasets or metrics could explain it | "X beats Y on images" vs "Y beats X on text" |
| **0** | No contradiction: compatible, unrelated, or simply different topics | "X reaches 95% on D1" vs "We survey methods for D2" |

Judge the two findings. Use the titles only for context.

**You do not need numbers or ground truth.** You are not deciding which paper is right; that would need full text. You are judging whether the two papers' *stated claims* conflict. Abstracts state claims in words ("X outperforms Y", "X fails to generalize", "no significant improvement").

- **Different numbers on different datasets or systems are never a contradiction on their own.** "Model A reaches 99% on dataset D1" vs "Model B reaches 98% on dataset D2" is **0**.
- Two papers that both report good results for their own methods is **0**.
- **2** needs the *same* subject (same method, or the same general claim) and *opposite* conclusions in words: "pre-training helps low-resource translation" vs "pre-training gives no benefit for low-resource translation".
- **1** is opposite-leaning claims under different conditions, where both could be true.
- If a sentence is too vague to compare, choose **0** and write "vague" in `notes`.

Most rows are expected to be 0. That is a result, not a problem: the sheet measures how many of the system's candidates are real contradictions.

## Task 6: OpenAlex API key (project owner)

Semantic Scholar hides reference lists for many publishers, and 77% of our papers had none. OpenAlex (CC0 open data) has most of them, matched by DOI. Keyless access shares a small daily budget that is often exhausted.

1. Create a free account at https://openalex.org and generate an API key from your account settings.
2. Add a line `OPENALEX_API_KEY=<your key>` to the ignored `.env`. Never commit it.

---

## Scoring (after sheets are returned)

```powershell
# Relevance: human-human and human-AI agreement
.\venv\Scripts\python.exe -m src.evaluation.human_agreement --sheets evaluation\human_review\relevance_sheet_cv.csv evaluation\human_review\relevance_sheet_<initials>.csv
# Relevance subset (task 3b): kappa vs CV and vs the 7B / 27B judges on the same 50 pairs
.\venv\Scripts\python.exe evaluation\human_review\build_relevance_subset.py score --sheet evaluation\human_review\relevance_subset50_sheet_<initials>.csv
# Hypotheses: per-arm means, graph minus vector (paired by query), inter-rater kappa
.\venv\Scripts\python.exe -m src.evaluation.human_ratings hypothesis --sheets evaluation\human_review\hypothesis_sheet_<A>.csv evaluation\human_review\hypothesis_sheet_<B>.csv
# Contradictions: precision of top-ranked candidates vs lower-ranked controls, inter-rater kappa
.\venv\Scripts\python.exe -m src.evaluation.human_ratings contradiction --sheets evaluation\human_review\contradiction_sheet_<A>.csv evaluation\human_review\contradiction_sheet_<B>.csv
```

```powershell
# Round 2
.\venv\Scripts\python.exe -m src.evaluation.human_ratings hypothesis_round2 --sheets evaluation\human_review\hypothesis_round2_sheet_<A>.csv
.\venv\Scripts\python.exe -m src.evaluation.human_ratings contradiction_round2 --sheets evaluation\human_review\contradiction_round2_sheet_<A>.csv
```

## Rater record

All raters are anonymous. Publications and reports refer to them only as Rater A, B and C, with their role; no names are recorded. The suffixes in sheet file names (`_cv`, `_rater3`) are internal sheet IDs, not identities; for double-blind submission, share an anonymised copy of the repository.

| Rater | Sheets | Role | Relation to the system | AI tools used | Saw other raters' sheets |
|---|---|---|---|---|---|
| Rater A (sheets `*_cv.csv`) | relevance 200, hypotheses R1 and R2, contradictions R1 and R2 | Project member | Developer of the system; blind to method and AI grade on every sheet (method keys hidden) | None | n/a (rated first) |
| Rater B (sheets `*_rater3.csv`) | relevance subset 50, hypotheses R1, contradictions R1 | Project member | Did not build the system (confirmed by the project owner, 2026-11-19) | None (confirmed by the project owner, 2026-11-19) | No (confirmed by the project owner, 2026-11-19) |
| Rater C (task 9, judge transfer 100) | judge transfer 100 | ___ (role only) | ___ | ___ | ___ |

Known limits to state in the paper: Rater A is the developer and rates every task; round 2 has one rater; Rater B is a project member, though not a developer, and their independence rests on the owner's confirmation, not a recorded protocol.

## Sample designs

- **Relevance:** 200 query–paper pairs from the frozen 1,145-label GNN pool, stratified by the original 3B AI grade (80/50/70 for grades 0/1/2). Seed 20261114 (`build_relevance_sheet.py`).
- **Hypotheses and contradictions:** 30 of the 44 stratified queries chosen by hash, seed 20261114 (`build_review_kit.py`). The selected configuration was used: BGE retrieval, BM25, graph expansion, and the SciERC entity graph.
  - Hypothesis arms per query: the top structural-hole candidate from the entity graph, and a vector-neighbour baseline (vector rank 6–15, not cited by the top 5). Rows are shuffled, and the linking concepts stay in the key because they would reveal the arm.
  - Contradiction rows: the top two NLI-ranked candidates among the top 20 retrieved papers (56 rows), plus a lower-ranked control (rank 5–15) for 12 queries.
- **Round 2 (`build_round2_kit.py`, seed 20261115):** the 14 stratified queries not used in round 1.
  - Hypothesis arms per query: vector neighbour, original structural hole, problem-bridge (margin 0.10, selected on the round-1 queries with the validated 7B relevance judge), and problem-bridge restricted to specific entities (tied on dev, secondary). A paper chosen by several arms appears once.
  - Contradiction pairs come from the corpus-wide citation-grounded index: all 9 the verdict model called contradictions, 15 unflagged from the top 150, and 16 from ranks 151+.
