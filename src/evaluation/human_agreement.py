"""Agreement between human relevance grades and the AI judge.

Reads completed copies of ``evaluation/human_review/relevance_sheet_blank.csv``
and the hidden key, then reports human-human and human-AI agreement with
quadratic-weighted Cohen's kappa on the 0/1/2 scale.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

GRADES = (0, 1, 2)
KEY = "evaluation/human_review/relevance_key_DO_NOT_SHARE_WITH_ANNOTATORS.json"


def weighted_kappa(first, second, categories=GRADES):
    """Quadratic-weighted Cohen's kappa."""
    if len(first) != len(second) or not first:
        raise ValueError("ratings must be non-empty and aligned")
    index = {value: i for i, value in enumerate(categories)}
    size = len(categories)
    observed = [[0.0] * size for _ in range(size)]
    for a, b in zip(first, second):
        observed[index[a]][index[b]] += 1
    total = len(first)
    rows = [sum(r) for r in observed]
    cols = [sum(observed[i][j] for i in range(size)) for j in range(size)]
    weight = lambda i, j: ((i - j) ** 2) / ((size - 1) ** 2)
    observed_disagreement = sum(weight(i, j) * observed[i][j] / total for i in range(size) for j in range(size))
    expected_disagreement = sum(weight(i, j) * rows[i] * cols[j] / total ** 2 for i in range(size) for j in range(size))
    return round(1 - observed_disagreement / expected_disagreement, 4) if expected_disagreement else None


def pair_report(first, second):
    exact = sum(a == b for a, b in zip(first, second)) / len(first)
    binary = sum((a > 0) == (b > 0) for a, b in zip(first, second)) / len(first)
    return {"n": len(first), "quadratic_weighted_kappa": weighted_kappa(first, second),
            "exact_agreement": round(exact, 4), "binary_relevant_agreement": round(binary, 4)}


def read_sheet(path):
    grades = {}
    with open(path, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            value = row["grade_0_1_2"].strip()
            if value not in {"0", "1", "2"}:
                raise ValueError(f"{path}: item {row['item_id']} has grade {value!r}; expected 0, 1 or 2")
            grades[row["item_id"]] = int(value)
    return grades


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheets", nargs="+", required=True)
    parser.add_argument("--key", default=KEY)
    parser.add_argument("--output", default="evaluation/human_review/agreement_report.json")
    args = parser.parse_args(argv)
    key = {item["item_id"]: item["ai_grade"] for item in json.loads(Path(args.key).read_text())["items"]}
    sheets = {Path(path).stem: read_sheet(path) for path in args.sheets}
    for name, grades in sheets.items():
        if set(grades) != set(key):
            raise ValueError(f"{name} does not cover exactly the sampled items")
    items = sorted(key)
    report = {"items": len(items), "human_vs_ai": {}, "human_vs_human": {}}
    names = sorted(sheets)
    for name in names:
        report["human_vs_ai"][name] = pair_report([sheets[name][i] for i in items], [key[i] for i in items])
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            report["human_vs_human"][f"{first}~{second}"] = pair_report(
                [sheets[first][x] for x in items], [sheets[second][x] for x in items])
    if len(names) >= 2:
        agreed = [x for x in items if len({sheets[n][x] for n in names}) == 1]
        consensus = [sheets[names[0]][x] for x in agreed]
        report["ai_vs_consensus"] = {**pair_report(consensus, [key[x] for x in agreed]),
                                     "consensus_items": len(agreed),
                                     "confusion_human_rows_ai_columns": [
                                         [sum(1 for x, c in zip(agreed, consensus) if c == h and key[x] == a)
                                          for a in GRADES] for h in GRADES]}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
