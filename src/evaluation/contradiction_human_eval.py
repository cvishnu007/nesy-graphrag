"""Run a contradiction verdict model on human-labelled in-corpus candidate pairs.

The human sheet (``evaluation/human_review/contradiction_sheet_*.csv``) labels
each NLI-screened pair 0 (none), 1 (tension), or 2 (contradiction). This module
asks a verdict model the same question with the application's structured prompt
(both abstracts plus the two focus sentences) and reports how its verdicts fall
against the human labels, in particular the false-alarm rate on pairs humans
labelled 0. With few or no human positives, recall cannot be estimated here;
use SciFact (``contradiction_verdict_eval.py``) for that.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

KEY = "evaluation/human_review/contradiction_key_DO_NOT_SHARE_WITH_ANNOTATORS.json"


def load_pairs(sheet, key_path=KEY, corpus=None):
    from src.utils.config import CLEAN_FILE

    key = {item["item_id"]: item for item in json.loads(Path(key_path).read_text())["items"]}
    corpus = corpus or {p["id"]: p for p in json.loads(Path(CLEAN_FILE).read_text(encoding="utf-8"))}
    pairs = []
    with open(sheet, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            hidden = key[row["item_id"]]
            papers = []
            for side in ("1", "2"):
                paper = corpus[hidden[f"paper_{side}_id"]]
                papers.append({"title": paper.get("clean_title", paper.get("title", "")),
                               "year": paper.get("year", ""),
                               "abstract": paper.get("clean_abstract", paper.get("abstract", ""))})
            pairs.append({"item_id": row["item_id"], "human": int(row["label_0_1_2"]), "arm": hidden["arm"],
                          "nli": hidden["nli_contradiction"], "papers": papers,
                          "evidence": ({"text": row["finding_1"]}, {"text": row["finding_2"]})})
    return pairs


def verdict(client, pair):
    from src.pipeline.prompts import build_contradiction_prompt
    from src.pipeline.verdicts import CONTRADICTION_JSON_SCHEMA, parse_contradiction_response

    prompt = build_contradiction_prompt(*pair["papers"], structured=True, abstract_chars=1500,
                                        evidence=pair["evidence"])
    options = {"json_schema": CONTRADICTION_JSON_SCHEMA} if client.provider == "ollama" else {}
    if client.provider == "groq" and "gpt-oss" in client.model:
        options["reasoning_effort"] = "low"
    raw = client.complete(prompt, max_tokens=1500, temperature=0.0, **options)
    return raw, parse_contradiction_response(raw)


def report(pairs, rows):
    by_item = {row["item_id"]: row for row in rows}
    table = Counter((pair["human"], by_item[pair["item_id"]]["verdict"]) for pair in pairs)
    human_negative = [p for p in pairs if p["human"] == 0]
    flagged = [p for p in pairs if by_item[p["item_id"]]["verdict"] == "CONTRADICTION"]
    return {
        "pairs": len(pairs),
        "human_label_counts": dict(Counter(p["human"] for p in pairs)),
        "verdict_counts": dict(Counter(r["verdict"] for r in rows)),
        "human_label_x_verdict": {f"{h}|{v}": n for (h, v), n in sorted(table.items())},
        "false_alarm_rate_on_human_0": round(sum(by_item[p["item_id"]]["verdict"] == "CONTRADICTION"
                                                 for p in human_negative) / max(1, len(human_negative)), 4),
        "flagged": len(flagged),
        "flagged_with_human_label_ge1": sum(p["human"] >= 1 for p in flagged),
        "valid_rate": round(sum(r["valid"] for r in rows) / len(rows), 4),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", default="evaluation/human_review/contradiction_sheet_labeled_cv.csv")
    parser.add_argument("--llm", required=True, help="ollama:<model> or groq:<model>")
    parser.add_argument("--output-dir", default="results/reasoning/contradiction_human_pairs_20261114")
    args = parser.parse_args(argv)
    from src.evaluation.judge_agreement import make_client

    pairs = load_pairs(args.sheet)
    client = make_client(args.llm)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", args.llm)
    checkpoint = output / f"{safe}.jsonl"
    done = {}
    if checkpoint.exists():
        done = {r["item_id"]: r for r in map(json.loads, checkpoint.read_text(encoding="utf-8").splitlines())
                if not r["raw"].startswith("ERROR")}
    with checkpoint.open("a", encoding="utf-8") as stream:
        for pair in pairs:
            if pair["item_id"] in done:
                continue
            try:
                raw, parsed = verdict(client, pair)
            except Exception as error:  # a failed call is retried on resume, never scored
                raw, parsed = f"ERROR {type(error).__name__}", {"verdict": "UNKNOWN", "confidence": None, "valid": False}
            row = {"item_id": pair["item_id"], "verdict": parsed["verdict"], "confidence": parsed["confidence"],
                   "valid": parsed["valid"], "raw": raw[:2000]}
            done[pair["item_id"]] = row
            stream.write(json.dumps(row) + "\n")
            stream.flush()
    rows = [done[p["item_id"]] for p in pairs]
    result = {"llm": args.llm, "sheet": args.sheet, **report(pairs, rows)}
    summary = output / "summary.json"
    data = json.loads(summary.read_text()) if summary.exists() else {}
    data[args.llm] = result
    summary.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
