"""Score completed hypothesis and contradiction rating sheets against their hidden keys.

Hypothesis sheets: mean 1-5 ratings per arm (graph structural hole vs vector
neighbour), paired by query with an exact randomization test, plus
inter-rater quadratic-weighted kappa when two or more sheets are given.

Contradiction sheets: precision of the top-ranked NLI candidates (label 2 =
real contradiction; label >= 1 = contradiction or tension) vs lower-ranked
controls, plus inter-rater kappa.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from src.evaluation.human_agreement import weighted_kappa
from src.evaluation.significance import paired_permutation_test, paired_significance

HYPOTHESIS_FIELDS = ("relevance_1_5", "novelty_1_5", "promise_1_5")


def read(path, fields, allowed):
    rows = {}
    with open(path, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            values = {}
            for field in fields:
                value = row[field].strip()
                if value not in allowed:
                    raise ValueError(f"{path}: {row['item_id']} {field}={value!r}; expected one of {sorted(allowed)}")
                values[field] = int(value)
            rows[row["item_id"]] = values
    return rows


def kappas(sheets, fields, categories):
    names = sorted(sheets)
    result = {}
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            items = sorted(set(sheets[first]) & set(sheets[second]))
            result[f"{first}~{second}"] = {
                field: weighted_kappa([sheets[first][x][field] for x in items],
                                      [sheets[second][x][field] for x in items], categories) for field in fields}
    return result


def mean_ratings(sheets):
    """Average raters per item."""
    items = set.intersection(*(set(s) for s in sheets.values()))
    return {x: {f: fmean(s[x][f] for s in sheets.values()) for f in next(iter(sheets.values()))[x]} for x in items}


def hypothesis_report(sheets, key):
    ratings = mean_ratings(sheets)
    by_query = defaultdict(dict)
    for item in key["items"]:
        if item["item_id"] in ratings:
            by_query[item["query_id"]][item["arm"]] = ratings[item["item_id"]]
    report = {"raters": sorted(sheets), "inter_rater_kappa": kappas(sheets, HYPOTHESIS_FIELDS, (1, 2, 3, 4, 5)),
              "arms": {}, "graph_minus_vector": {}}
    for arm in ("graph_structural_hole", "vector_neighbour"):
        values = [q[arm] for q in by_query.values() if arm in q]
        report["arms"][arm] = {"n": len(values), **{f: round(fmean(v[f] for v in values), 3) for f in HYPOTHESIS_FIELDS}}
    paired = [q for q in by_query.values() if len(q) == 2]
    for field in HYPOTHESIS_FIELDS:
        deltas = [q["graph_structural_hole"][field] - q["vector_neighbour"][field] for q in paired]
        report["graph_minus_vector"][field] = (paired_significance(deltas) if len(deltas) <= 20
                                               else paired_permutation_test(deltas))
    return report


ROUND2_ARMS = ("vector_neighbour", "graph_structural_hole", "bridge_m10", "bridge_m10_specific")
ROUND2_COMPARISONS = (("bridge_m10", "vector_neighbour"), ("bridge_m10", "graph_structural_hole"),
                      ("bridge_m10_specific", "vector_neighbour"), ("graph_structural_hole", "vector_neighbour"))


def hypothesis_round2_report(sheets, key):
    """An item chosen by several arms counts for each of them; comparisons are paired by query."""
    ratings = mean_ratings(sheets)
    by_query = defaultdict(dict)
    for item in key["items"]:
        if item["item_id"] in ratings:
            for arm in item["arms"]:
                by_query[item["query_id"]][arm] = ratings[item["item_id"]]
    report = {"raters": sorted(sheets), "inter_rater_kappa": kappas(sheets, HYPOTHESIS_FIELDS, (1, 2, 3, 4, 5)),
              "arms": {}, "comparisons": {}}
    for arm in ROUND2_ARMS:
        values = [q[arm] for q in by_query.values() if arm in q]
        if values:
            report["arms"][arm] = {"n": len(values), **{f: round(fmean(v[f] for v in values), 3)
                                                        for f in HYPOTHESIS_FIELDS}}
    for challenger, reference in ROUND2_COMPARISONS:
        paired = [q for q in by_query.values() if challenger in q and reference in q]
        if paired:
            report["comparisons"][f"{challenger}_minus_{reference}"] = {
                field: paired_significance([q[challenger][field] - q[reference][field] for q in paired])
                for field in HYPOTHESIS_FIELDS}
    return report


def contradiction_report(sheets, key):
    ratings = mean_ratings(sheets)
    report = {"raters": sorted(sheets), "inter_rater_kappa": kappas(sheets, ("label_0_1_2",), (0, 1, 2)), "arms": {}}
    for arm in sorted({i["arm"] for i in key["items"]}):
        labels = [ratings[i["item_id"]]["label_0_1_2"] for i in key["items"] if i["arm"] == arm and i["item_id"] in ratings]
        if labels:
            report["arms"][arm] = {"n": len(labels),
                                   "precision_contradiction": round(sum(v >= 1.5 for v in labels) / len(labels), 3),
                                   "precision_contradiction_or_tension": round(sum(v >= 0.5 for v in labels) / len(labels), 3)}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=("hypothesis", "contradiction", "hypothesis_round2", "contradiction_round2"))
    parser.add_argument("--sheets", nargs="+", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    key = json.loads(Path(f"evaluation/human_review/{args.kind}_key_DO_NOT_SHARE_WITH_ANNOTATORS.json").read_text())
    if args.kind.startswith("hypothesis"):
        sheets = {Path(p).stem: read(p, HYPOTHESIS_FIELDS, {"1", "2", "3", "4", "5"}) for p in args.sheets}
        report = (hypothesis_round2_report if args.kind.endswith("round2") else hypothesis_report)(sheets, key)
    else:
        sheets = {Path(p).stem: read(p, ("label_0_1_2",), {"0", "1", "2"}) for p in args.sheets}
        report = contradiction_report(sheets, key)
    text = json.dumps(report, indent=2)
    Path(args.output or f"evaluation/human_review/{args.kind}_report.json").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
