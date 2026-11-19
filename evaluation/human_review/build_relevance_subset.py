"""Blinded 50-item relevance subset for an independent second grader, plus its scorer.

Build (deterministic, seed 20261119):
    python evaluation/human_review/build_relevance_subset.py build
Score a completed copy:
    python evaluation/human_review/build_relevance_subset.py score --sheet <completed.csv>

Items come only from the held-out half of the 200-pair sheet (split "judge-selection-v1"),
so the 7B and 27B judges can be compared on the same pairs. Sampling is stratified by the
first grader's grade (20 x 0, 12 x 1, 18 x 2) so every grade has enough items for kappa.
Items are renumbered S01..S50 and shuffled, so the sheet cannot be lined up with
relevance_sheet_cv.csv or any other graded copy. The S -> R mapping lives in the
gitignored *_DO_NOT_SHARE_* key.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.human_agreement import pair_report  # noqa: E402
from src.evaluation.judge_agreement import split_items  # noqa: E402

HERE = Path("evaluation/human_review")
SOURCE = HERE / "relevance_sheet_blank.csv"
FIRST_GRADER = HERE / "relevance_sheet_cv.csv"
SHEET = HERE / "relevance_subset50_sheet_blank.csv"
KEY = HERE / "relevance_subset50_key_DO_NOT_SHARE_WITH_ANNOTATORS.json"
QUOTA = {0: 20, 1: 12, 2: 18}
JUDGES = {
    "7b_strict_batch1": "results/retrieval/judge_agreement_local_20261114/held_out_ollama_qwen2_5_7b_instruct_strict_b1.jsonl",
    "27b_strict": "results/retrieval/judge_agreement_groq_20261114/held_out_groq_qwen_qwen3_8_27b_strict.jsonl",
}


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def build():
    source = {r["item_id"]: r for r in rows(SOURCE)}
    first = {r["item_id"]: int(r["grade_0_1_2"]) for r in rows(FIRST_GRADER)}
    _, held_out = split_items(list(source))
    rng = random.Random(20261119)
    picked = []
    for grade, n in QUOTA.items():
        picked += rng.sample(sorted(i for i in held_out if first[i] == grade), n)
    rng.shuffle(picked)
    sheet, key = [], []
    for n, item in enumerate(picked, 1):
        r = source[item]
        sheet.append({"item_id": f"S{n:02d}", "query": r["query"], "paper_title": r["paper_title"],
                      "paper_abstract": r["paper_abstract"], "grade_0_1_2": "", "notes": ""})
        key.append({"item_id": f"S{n:02d}", "source_item_id": item})
    with open(SHEET, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sheet[0]))
        writer.writeheader()
        writer.writerows(sheet)
    KEY.write_text(json.dumps({"seed": 20261119, "source": str(SOURCE), "split": "judge-selection-v1 held_out",
                               "quota_by_first_grader": QUOTA, "items": key}, indent=2) + "\n")
    print(f"wrote {SHEET} ({len(sheet)} items) and {KEY}")


def score(sheet_path, output):
    mapping = {k["item_id"]: k["source_item_id"] for k in json.loads(KEY.read_text())["items"]}
    graded = {}
    for r in rows(sheet_path):
        value = r["grade_0_1_2"].strip()
        if value not in {"0", "1", "2"}:
            raise ValueError(f"{sheet_path}: {r['item_id']} has grade {value!r}; expected 0, 1 or 2")
        graded[mapping[r["item_id"]]] = int(value)
    if set(graded) != set(mapping.values()):
        raise ValueError("sheet does not cover exactly the 50 subset items")
    items = sorted(graded)
    first = {r["item_id"]: int(r["grade_0_1_2"]) for r in rows(FIRST_GRADER)}
    second = [graded[i] for i in items]
    report = {"items": len(items), "second_vs_first_grader": pair_report([first[i] for i in items], second),
              "judges": {}}
    agreed = [i for i in items if first[i] == graded[i]]
    for name, path in JUDGES.items():
        judge = {r["item_id"]: r["grade"] for r in map(json.loads, Path(path).read_text().splitlines())}
        report["judges"][name] = {
            "vs_first_grader": pair_report([first[i] for i in items], [judge[i] for i in items]),
            "vs_second_grader": pair_report(second, [judge[i] for i in items]),
            "vs_consensus": {**pair_report([first[i] for i in agreed], [judge[i] for i in agreed]),
                             "consensus_items": len(agreed)},
        }
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    s = sub.add_parser("score")
    s.add_argument("--sheet", required=True)
    s.add_argument("--output", default=str(HERE / "relevance_subset50_report.json"))
    args = parser.parse_args(argv)
    build() if args.command == "build" else score(args.sheet, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
