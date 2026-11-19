"""Blinded sheet to check that the 7B relevance judge transfers beyond GNN queries, plus its scorer.

The judge was validated on 200 pairs from the GNN benchmark (RESULTS 2b). The headline
44-query benchmark (RESULTS 2d) spans 11 topics, so this sheet samples its pool from the
10 non-GNN topics: 10 pairs per topic, 4 / 2 / 4 by the judge's grade 0 / 1 / 2 (grade 1 is
oversampled, as in the first sheet, so every grade has enough items for kappa).

Build (deterministic, seed 20261120; needs the cleaned corpus for paper text):
    python evaluation/human_review/build_judge_transfer_sheet.py build
Grade the same 100 pairs with another judge (one paper per call; saved under JUDGE_DIR):
    python evaluation/human_review/build_judge_transfer_sheet.py judge --judge groq:qwen/qwen3.8-27b@strict
Score a completed copy (needs only committed files; also scores every saved extra judge):
    python evaluation/human_review/build_judge_transfer_sheet.py score --sheet <completed.csv>

The sheet shows only query, title and abstract; items are numbered T001..T100 in shuffled
order. The judge grades are in the gitignored *_DO_NOT_SHARE_* key, and the T -> pair mapping
can be rebuilt from committed files with picked_items().
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.human_agreement import pair_report  # noqa: E402

HERE = Path("evaluation/human_review")
QUERIES = Path("evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json")
LABELS = Path("results/retrieval/stratified_v2_20261114/labels.jsonl")
SHEET = HERE / "judge_transfer_sheet_blank.csv"
KEY = HERE / "judge_transfer_key_DO_NOT_SHARE_WITH_ANNOTATORS.json"
REPORT = HERE / "judge_transfer_report.json"
JUDGE_DIR = Path("results/retrieval/judge_transfer_20261120")
EXCLUDED_TOPIC = "graph neural networks"  # already covered by the first validation
QUOTA = {0: 4, 1: 2, 2: 4}
SEED = 20261120
FIRST_VALIDATION_KAPPA = 0.75  # held-out, GNN pairs (RESULTS 2b)
VARIANTS = ("production_absolute", "bge", "bm25", "rrf_bge_bm25", "full_bge_bm25_similar")


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def judge_labels():
    return {r["item_id"]: int(r["grade"]) for r in map(json.loads, LABELS.read_text().splitlines())}


def queries():
    return {q["query_id"]: q for q in json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]}


def picked_items():
    """The 100 pool item ids ("<query_id>/<paper_id>") in sheet order T001..T100."""
    labels, qs = judge_labels(), queries()
    by_topic = defaultdict(lambda: defaultdict(list))
    for item, grade in labels.items():
        topic = qs[item.split("/")[0]]["topic"]
        if topic != EXCLUDED_TOPIC:
            by_topic[topic][grade].append(item)
    rng = random.Random(SEED)
    picked = []
    for topic in sorted(by_topic):
        quota = dict(QUOTA)
        # A topic short of one grade gives the missing items to the other grades, larger first.
        for grade in (1, 2, 0):
            short = quota[grade] - len(by_topic[topic][grade])
            if short > 0:
                quota[grade] -= short
                for other in (g for g in (2, 0, 1) if g != grade):
                    spare = len(by_topic[topic][other]) - quota[other]
                    take = min(short, max(spare, 0))
                    quota[other] += take
                    short -= take
        for grade, n in quota.items():
            picked += rng.sample(sorted(by_topic[topic][grade]), n)
    rng.shuffle(picked)
    return picked


def subset_mapping():
    return {f"T{n:03d}": item for n, item in enumerate(picked_items(), 1)}


def build():
    from src.utils.config import CLEAN_FILE

    corpus = {p["id"]: p for p in json.loads(Path(CLEAN_FILE).read_text(encoding="utf-8"))}
    labels, qs = judge_labels(), queries()
    sheet, key = [], []
    for item_id, item in subset_mapping().items():
        qid, pid = item.split("/", 1)
        paper = corpus[pid]
        sheet.append({"item_id": item_id, "query": qs[qid]["query"],
                      "paper_title": paper.get("clean_title", paper.get("title", "")),
                      "paper_abstract": paper.get("clean_abstract", paper.get("abstract", "")),
                      "grade_0_1_2": "", "notes": ""})
        key.append({"item_id": item_id, "source_item_id": item, "topic": qs[qid]["topic"],
                    "split": qs[qid]["split"], "judge_grade": labels[item]})
    with open(SHEET, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sheet[0]))
        writer.writeheader()
        writer.writerows(sheet)
    KEY.write_text(json.dumps({"seed": SEED, "source": str(LABELS), "quota_per_topic_by_judge_grade": QUOTA,
                               "excluded_topic": EXCLUDED_TOPIC, "items": key}, indent=2) + "\n")
    print(f"wrote {SHEET} ({len(sheet)} items) and {KEY}")


def run_judge(spec):
    """Grade the blank sheet's 100 pairs with ``provider:model@prompt``, one paper per call."""
    import re
    from datetime import date

    from src.evaluation.judge_agreement import judge, make_client

    model_spec, prompt_name = spec.rsplit("@", 1)
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", spec)
    sheet = [{"item_id": r["item_id"], "query": r["query"], "paper_title": r["paper_title"],
              "paper_abstract": r["paper_abstract"]} for r in rows(SHEET)]
    grades = judge(make_client(model_spec), sheet, prompt_name, JUDGE_DIR / f"{safe}.jsonl", batch_size=1)
    meta = JUDGE_DIR / "runs.json"
    runs = json.loads(meta.read_text()) if meta.exists() else {}
    runs[safe] = {"judge": spec, "papers_per_call": 1, "temperature": 0.0, "date": date.today().isoformat(),
                  "items": len(grades)}
    meta.write_text(json.dumps(runs, indent=2) + "\n")
    print(f"graded {len(grades)} items with {spec}")


def extra_judges():
    """Saved extra-judge grades: {name: {T-id: grade}}."""
    runs = json.loads((JUDGE_DIR / "runs.json").read_text()) if (JUDGE_DIR / "runs.json").exists() else {}
    return {name: {r["item_id"]: int(r["grade"]) for r in map(json.loads, (JUDGE_DIR / f"{name}.jsonl").read_text().splitlines())}
            for name in runs}


def score(sheet_path, output=REPORT):
    mapping, labels, qs = subset_mapping(), judge_labels(), queries()
    human = {}
    for r in rows(sheet_path):
        value = r["grade_0_1_2"].strip()
        if value not in {"0", "1", "2"}:
            raise ValueError(f"{sheet_path}: {r['item_id']} has grade {value!r}; expected 0, 1 or 2")
        human[mapping[r["item_id"]]] = int(value)
    if set(human) != set(mapping.values()):
        raise ValueError(f"sheet does not cover exactly the {len(mapping)} items")
    items = sorted(human)
    overall = pair_report([human[i] for i in items], [labels[i] for i in items])
    by_topic = defaultdict(list)
    for i in items:
        by_topic[qs[i.split("/")[0]]["topic"]].append(i)
    per_topic = {}
    for topic, its in sorted(by_topic.items()):
        h, j = [human[i] for i in its], [labels[i] for i in its]
        per_topic[topic] = {"items": len(its), "exact_agreement": round(sum(a == b for a, b in zip(h, j)) / len(its), 3),
                            "mean_human": round(sum(h) / len(h), 3), "mean_judge": round(sum(j) / len(j), 3)}
    # Does the judge's error depend on which system retrieved the paper? Items in each
    # variant's top 10 (RESULTS 2d rankings); mean human and judge grade on those items.
    rankings = defaultdict(dict)
    for row in map(json.loads, (LABELS.parent / "rankings.jsonl").read_text().splitlines()):
        rankings[row["variant"]][row["query_id"]] = row["ranked_paper_ids"][:10]
    by_variant = {}
    extras = extra_judges()
    t_of = {item: t for t, item in mapping.items()}
    for variant in VARIANTS:
        its = [i for i in items if i.split("/", 1)[1] in rankings[variant].get(i.split("/")[0], [])]
        if its:
            h, j = [human[i] for i in its], [labels[i] for i in its]
            by_variant[variant] = {"items_in_top10": len(its), "mean_human": round(sum(h) / len(h), 3),
                                   "mean_judge": round(sum(j) / len(j), 3),
                                   "human_minus_judge": round((sum(h) - sum(j)) / len(h), 3)}
            for name, grades in extras.items():
                g = [grades[t_of[i]] for i in its]
                by_variant[variant][f"human_minus_{name}"] = round((sum(h) - sum(g)) / len(h), 3)
    by_t = {t: human[item] for t, item in mapping.items()}
    other = {}
    for name, grades in extra_judges().items():
        ts = sorted(by_t)
        other[name] = pair_report([by_t[t] for t in ts], [grades[t] for t in ts])
        other[name]["mean_judge"] = round(sum(grades[t] for t in ts) / len(ts), 3)
    report = {"sheet": str(sheet_path), "judge": "ollama:qwen2.5:7b-instruct@strict (stored pool labels)",
              "items": len(items), "judge_vs_human": overall, "other_judges_vs_human": other,
              "per_topic": per_topic,
              "by_variant_top10": by_variant,
              "first_validation_heldout_kappa_gnn": FIRST_VALIDATION_KAPPA,
              "threshold": "predeclared: quadratic-weighted kappa >= 0.6, as for the first validation",
              "note": "Sample stratified by judge grade, so kappa is not directly comparable with a random sample."}
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("items", "judge_vs_human", "other_judges_vs_human")}, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("build", "judge", "score"))
    parser.add_argument("--sheet")
    parser.add_argument("--judge", help="provider:model@prompt for the judge stage")
    parser.add_argument("--output", default=str(REPORT))
    args = parser.parse_args(argv)
    if args.stage == "build":
        build()
    elif args.stage == "judge":
        if not args.judge:
            parser.error("judge needs --judge")
        run_judge(args.judge)
    else:
        if not args.sheet:
            parser.error("score needs --sheet")
        score(Path(args.sheet), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
