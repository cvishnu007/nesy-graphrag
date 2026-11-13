# Human Relevance Review

Purpose: measure how well the AI relevance judge (local `qwen2.5:3b-instruct`) agrees with people. Every retrieval score in the project rests on that judge, so this is the most important validation step for publication.

## What to do (about 2–3 hours per person)

1. Two people work **independently**. Each copies `relevance_sheet_blank.csv` to `relevance_sheet_<initials>.csv` (for example `relevance_sheet_cv.csv`). Do not discuss items until both are finished.
2. For each row, read the query, then the paper title and abstract, and enter a grade in `grade_0_1_2`:

| Grade | Meaning | Example for "graph neural networks for fraud detection" |
|---|---|---|
| **2** | Directly about the query topic | A GNN model for detecting financial fraud |
| **1** | Related, useful, but not directly about it | A GNN for general anomaly detection; a non-GNN fraud detector |
| **0** | Not about the query | A GNN for molecules; a survey of blockchain |

3. Judge only from the title and abstract. Do not search the web. When unsure between two grades, choose the lower one and write a short note.
4. Leave the other columns unchanged. Save as CSV (UTF-8).

Do **not** open `relevance_key_DO_NOT_SHARE_WITH_ANNOTATORS.json`. It holds the AI grades; seeing them would bias the review.

## Scoring (run after both sheets are complete)

```powershell
.\venv\Scripts\python.exe -m src.evaluation.human_agreement `
  --sheets evaluation\human_review\relevance_sheet_A.csv evaluation\human_review\relevance_sheet_B.csv
```

It reports human–human agreement, each person's agreement with the AI judge (quadratic-weighted Cohen's kappa and exact agreement), and the AI judge's confusion matrix against the human consensus.

## Sample design

200 query–paper pairs from the frozen 1,145-label pool (all 20 queries), stratified by AI grade: 80 graded 0, 50 graded 1, 70 graded 2. Grade 1 is oversampled because the AI judge uses it rarely. Random seed 20261114.
