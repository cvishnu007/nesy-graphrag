"""Rebuild the blinded relevance sheet and its hidden key (deterministic, seed 20261114).

Run from the repository root:  python evaluation/human_review/build_relevance_sheet.py
The key file is gitignored so annotators with repository access cannot see AI grades.
"""
import csv
import json
import random
from collections import defaultdict

random.seed(20261114)
bench = json.load(open("results/retrieval/expanded_20260913/benchmark.json", encoding="utf-8"))
rows = {(r["query_id"], r["paper_id"]): r for r in csv.DictReader(
    open("results/retrieval/expanded_20260913/candidates.csv", encoding="utf-8", newline=""))}
by_grade = defaultdict(list)
for q in bench["queries"]:
    for pid, g in q["judgments"].items():
        if (q["query_id"], pid) in rows:
            by_grade[g].append((q["query_id"], pid, g))
quota = {0: 80, 1: 50, 2: 70}  # oversample grade 1, which the AI judge rarely uses
sample = [x for g, n in quota.items() for x in random.sample(by_grade[g], n)]
random.shuffle(sample)
sheet, key = [], []
for i, (qid, pid, g) in enumerate(sample, 1):
    r = rows[(qid, pid)]
    item = f"R{i:03d}"
    sheet.append({"item_id": item, "query": r["query"], "paper_title": r["title"],
                  "paper_abstract": r["abstract"], "grade_0_1_2": "", "notes": ""})
    key.append({"item_id": item, "query_id": qid, "paper_id": pid, "ai_grade": g, "split": r["split"]})
with open("evaluation/human_review/relevance_sheet_blank.csv", "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(sheet[0]))
    writer.writeheader()
    writer.writerows(sheet)
json.dump({"seed": 20261114, "quota_by_ai_grade": quota,
           "source": "results/retrieval/expanded_20260913/benchmark.json", "items": key},
          open("evaluation/human_review/relevance_key_DO_NOT_SHARE_WITH_ANNOTATORS.json", "w"), indent=1)
print(len(sheet), "items")
