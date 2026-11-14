"""Development comparison of hypothesis generators on the 30 human-rated queries.

Predeclared selection rule (2026-11-14, before any result was seen): among the
bridge configurations, choose the one with the highest mean judged relevance of
the top candidate over all 30 dev queries, counting a query without a candidate
as 0. Relevance is judged by the human-validated local judge (qwen2.5-7B, strict
prompt, one paper per call; held-out kappa 0.75). The two human-rated arms (old
structural hole, vector neighbour) are judged too, to check the judge reproduces
the human ordering. Novelty cannot be judged automatically; the selected
configuration goes to a human round on the 14 unused queries.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import fmean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CONFIGS = {
    "bridge_m05": {"margin": 0.05},
    "bridge_m10": {"margin": 0.10},
    "bridge_m10_specific": {"margin": 0.10, "specific_only": True},
}
KEY = "evaluation/human_review/hypothesis_key_DO_NOT_SHARE_WITH_ANNOTATORS.json"
SHEET = "evaluation/human_review/hypothesis_sheet_rated_cv.csv"
QUERIES = "evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json"
OUT = Path("results/reasoning/hypothesis_dev_20261114")


def main() -> int:
    from src.evaluation.judge_agreement import judge, make_client
    from src.pipeline.hypothesis_bridge import generate_bridged_hypotheses
    from src.storage.neo4j_store import get_driver
    from src.utils.config import CLEAN_FILE

    corpus = {p["id"]: p for p in json.loads(Path(CLEAN_FILE).read_text(encoding="utf-8"))}
    queries = {q["query_id"]: q["query"] for q in json.loads(Path(QUERIES).read_text(encoding="utf-8"))["queries"]}
    key = json.loads(Path(KEY).read_text())["items"]
    human = {r["item_id"]: r for r in csv.DictReader(open(SHEET, encoding="utf-8-sig"))}
    dev_ids = sorted({item["query_id"] for item in key})
    OUT.mkdir(parents=True, exist_ok=True)

    candidates = {}
    for item in key:
        candidates[(item["arm"], item["query_id"])] = {"id": item["paper_id"],
                                                        "human_relevance": int(human[item["item_id"]]["relevance_1_5"])}
    driver = get_driver()
    try:
        for name, options in CONFIGS.items():
            for query_id in dev_ids:
                found = generate_bridged_hypotheses(driver, queries[query_id], top_k=1, **options)
                if found:
                    candidates[(name, query_id)] = {"id": found[0]["id"], "detail": found[0]}
    finally:
        driver.close()

    rows = []
    for (arm, query_id), value in sorted(candidates.items()):
        paper = corpus[value["id"]]
        rows.append({"item_id": f"{arm}|{query_id}", "query": queries[query_id],
                     "paper_title": paper["clean_title"], "paper_abstract": paper["clean_abstract"]})
    grades = judge(make_client("ollama:qwen2.5:7b-instruct"), rows, "strict", OUT / "judge_labels.jsonl", batch_size=1)

    report = {"rule": __doc__.split("Predeclared selection rule")[1].split(".", 1)[0], "arms": {}}
    for arm in ["graph_structural_hole", "vector_neighbour", *CONFIGS]:
        judged = [grades[f"{arm}|{q}"] if (arm, q) in candidates else 0 for q in dev_ids]
        present = [q for q in dev_ids if (arm, q) in candidates]
        entry = {"coverage": len(present), "mean_judged_relevance_0_2": round(fmean(judged), 3),
                 "mean_judged_relevance_when_present": round(fmean(grades[f"{arm}|{q}"] for q in present), 3)
                 if present else None}
        if arm in ("graph_structural_hole", "vector_neighbour"):
            entry["mean_human_relevance_1_5"] = round(fmean(candidates[(arm, q)]["human_relevance"] for q in present), 3)
        report["arms"][arm] = entry
    eligible = {name: report["arms"][name]["mean_judged_relevance_0_2"] for name in CONFIGS}
    report["selected"] = max(sorted(eligible), key=lambda name: eligible[name])
    report["candidates"] = {f"{a}|{q}": v.get("detail", {"id": v["id"]}) for (a, q), v in candidates.items()
                            if a in CONFIGS}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "candidates"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
