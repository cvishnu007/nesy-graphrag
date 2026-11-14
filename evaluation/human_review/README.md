# Human Review Tasks

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
| 3 | Second relevance annotator, same 200 pairs | Teammate (not CV) | 2–3 h | ⏳ Needed for human–human agreement |
| 4 | Hypothesis ratings, 60 items | 2 people | ~1.5 h each | ⏳ |
| 5 | Contradiction labels, 68 pairs | 2 people | ~1 h each | ⏳ |
| 6 | Free OpenAlex API key | Project owner | 5 min | ⏳ Unblocks citation densification for papers whose references Semantic Scholar hides |

For tasks 4 and 5, two raters are needed for agreement. If only one person is available, the results are still useful but are reported as single-rater.

---

## Task 3: Second relevance annotator

Sheet: `relevance_sheet_blank.csv`. For each row, read the query, then the paper's title and abstract, and enter `grade_0_1_2`:

| Grade | Meaning | Example for "graph neural networks for fraud detection" |
|---|---|---|
| **2** | The paper's main contribution addresses the whole query | A GNN model for detecting financial fraud |
| **1** | Addresses part of the query, or the topic is secondary | A GNN for general anomaly detection; a non-GNN fraud detector |
| **0** | Does not address the query's core subject | A GNN for molecules; a survey of blockchain |

Shared words are not enough: "network traffic" is not "road traffic".

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

## Task 6: OpenAlex API key (project owner)

Semantic Scholar hides reference lists for many publishers, and 77% of our papers had none. OpenAlex (CC0 open data) has most of them, matched by DOI. Keyless access shares a small daily budget that is often exhausted.

1. Create a free account at https://openalex.org and generate an API key from your account settings.
2. Add a line `OPENALEX_API_KEY=<your key>` to the ignored `.env`. Never commit it.

---

## Scoring (after sheets are returned)

```powershell
# Relevance: human-human and human-AI agreement
.\venv\Scripts\python.exe -m src.evaluation.human_agreement --sheets evaluation\human_review\relevance_sheet_cv.csv evaluation\human_review\relevance_sheet_<initials>.csv
# Hypotheses: per-arm means, graph minus vector (paired by query), inter-rater kappa
.\venv\Scripts\python.exe -m src.evaluation.human_ratings hypothesis --sheets evaluation\human_review\hypothesis_sheet_<A>.csv evaluation\human_review\hypothesis_sheet_<B>.csv
# Contradictions: precision of top-ranked candidates vs lower-ranked controls, inter-rater kappa
.\venv\Scripts\python.exe -m src.evaluation.human_ratings contradiction --sheets evaluation\human_review\contradiction_sheet_<A>.csv evaluation\human_review\contradiction_sheet_<B>.csv
```

## Sample designs

- **Relevance:** 200 query–paper pairs from the frozen 1,145-label GNN pool, stratified by the original 3B AI grade (80/50/70 for grades 0/1/2). Seed 20261114 (`build_relevance_sheet.py`).
- **Hypotheses and contradictions:** 30 of the 44 stratified queries chosen by hash, seed 20261114 (`build_review_kit.py`). The selected configuration was used: BGE retrieval, BM25, graph expansion, and the SciERC entity graph.
  - Hypothesis arms per query: the top structural-hole candidate from the entity graph, and a vector-neighbour baseline (vector rank 6–15, not cited by the top 5). Rows are shuffled, and the linking concepts stay in the key because they would reveal the arm.
  - Contradiction rows: the top two NLI-ranked candidates among the top 20 retrieved papers (56 rows), plus a lower-ranked control (rank 5–15) for 12 queries.
