"""Rebuild the hidden arm keys of the finished hypothesis and contradiction sheets.

The keys (``*_key_DO_NOT_SHARE_WITH_ANNOTATORS.json``) record which method produced
each blinded row. They are gitignored, so arm-level results could not be re-scored
from a fresh clone. Rating is finished, and three of the four keys can be rebuilt
exactly from committed files, because the kit builders draw everything from fixed seeds:

* Round 1 (``build_review_kit.py``, seed 20261114): rows are appended per query in a
  fixed order (hypotheses: structural-hole arm, then vector arm; contradictions: the
  two top-ranked NLI pairs, then the optional control), and one ``random.Random``
  shuffles both lists. The only unknown is how many random draws the control-pair
  choices used before the shuffles; it is found by search and accepted only if every
  shuffled row's query matches the blank sheet, for both sheets at once.
* Round 2 contradictions (``build_round2_kit.py``, seed 20261115): the pairs are
  sampled from the committed citation-grounded index, so the whole draw is replayed;
  it is accepted only if both findings of every row match the blank sheet.
* Round 2 hypotheses cannot be rebuilt: a paper chosen by several arms appears once,
  and which arms chose it depends on the live graph. That key must come from the
  machine that built the kit.

The rebuilt keys reproduce the reported arm-level numbers exactly
(``src/evaluation/reproduce.py``). Run from the repository root to write them under
their usual (gitignored) names, so ``src/evaluation/human_ratings.py`` works on a
fresh clone:

    python evaluation/human_review/rebuild_arm_keys.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QUERIES = ROOT / "evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json"
INDEX = ROOT / "results/reasoning/contradiction_index_20261114/index.jsonl"
ROUND1_SEED = 20261114  # build_review_kit.SEED
ROUND1_QUERIES = 30  # build_review_kit.N_QUERIES
ROUND2_SEED = 20261115  # build_round2_kit.SEED
ROUND2_PAIRS = 40
MAX_OFFSET = 400


def _sheet(name):
    with open(HERE / name, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def round1_queries():
    """The 30 round-1 queries, as build_review_kit.pick_queries() chooses them."""
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
    ranked = sorted(queries, key=lambda q: hashlib.sha256(f"{ROUND1_SEED}|{q['query_id']}".encode()).hexdigest())
    return sorted(ranked[:ROUND1_QUERIES], key=lambda q: q["query_id"])


def _shuffled(rng, n):
    order = list(range(n))
    rng.shuffle(order)
    return order


def round1_keys():
    """Return (hypothesis_key, contradiction_key) in the builder's key format."""
    picked = round1_queries()
    text = {q["query_id"]: q["query"] for q in picked}
    hyp, con = _sheet("hypothesis_sheet_blank.csv"), _sheet("contradiction_sheet_blank.csv")
    per_query = Counter(row["query"] for row in con)
    h_items = [(q["query_id"], arm) for q in picked for arm in ("graph_structural_hole", "vector_neighbour")]
    c_items = [(q["query_id"], "top_ranked" if slot < 2 else "lower_ranked_control", slot + 1)
               for q in picked for slot in range(per_query[q["query"]])]
    if len(h_items) != len(hyp) or len(c_items) != len(con):
        raise ValueError("round-1 sheets do not have the expected rows")
    found = set()
    for offset in range(MAX_OFFSET):
        rng = random.Random(ROUND1_SEED)
        for _ in range(offset):  # the control-pair draws consumed one or more 32-bit outputs each
            rng.getrandbits(32)
        h_order = _shuffled(rng, len(h_items))
        if any(text[h_items[j][0]] != hyp[pos]["query"] for pos, j in enumerate(h_order)):
            continue
        c_order = _shuffled(rng, len(c_items))
        if all(text[c_items[j][0]] == con[pos]["query"] for pos, j in enumerate(c_order)):
            found.add((tuple(h_order), tuple(c_order)))
    if len(found) != 1:  # adjacent offsets can give the same permutation; anything else is ambiguous
        raise ValueError(f"round-1 shuffle not identified uniquely ({len(found)} candidates)")
    h_order, c_order = found.pop()
    hypothesis = {"seed": ROUND1_SEED, "rebuilt_by": "evaluation/human_review/rebuild_arm_keys.py",
                  "items": [{"item_id": hyp[pos]["item_id"], "query_id": h_items[j][0], "arm": h_items[j][1]}
                            for pos, j in enumerate(h_order)]}
    contradiction = {"seed": ROUND1_SEED, "rebuilt_by": "evaluation/human_review/rebuild_arm_keys.py",
                     "items": [{"item_id": con[pos]["item_id"], "query_id": c_items[j][0], "arm": c_items[j][1],
                                "slot": c_items[j][2]} for pos, j in enumerate(c_order)]}
    return hypothesis, contradiction


def round2_contradiction_key():
    """Replay build_round2_kit.py's contradiction draw; returns the key in the builder's format."""
    index = [json.loads(line) for line in INDEX.read_text(encoding="utf-8").splitlines()]
    flagged = [r for r in index if r["verdict"] == "CONTRADICTION"]
    top_other = [r for r in index if r["rank"] <= 150 and r["verdict"] != "CONTRADICTION"]
    lower = [r for r in index if r["rank"] > 150]
    remaining = ROUND2_PAIRS - len(flagged)
    rng = random.Random(ROUND2_SEED)
    picked = ([(r, "index_verdict_contradiction") for r in flagged]
              + [(r, "index_top150_unflagged") for r in rng.sample(top_other, min(len(top_other), remaining // 2))]
              + [(r, "index_rank151_plus") for r in rng.sample(lower, min(len(lower), remaining - remaining // 2))])
    hypotheses = _sheet("hypothesis_round2_sheet_blank.csv")
    _shuffled(rng, len(hypotheses))  # the hypothesis rows are shuffled first with the same generator
    rng.shuffle(picked)
    rows = _sheet("contradiction_round2_sheet_blank.csv")
    if len(rows) != len(picked) or any(
            r["evidence_1"]["text"] != row["finding_1"] or r["evidence_2"]["text"] != row["finding_2"]
            for (r, _), row in zip(picked, rows)):
        raise ValueError("round-2 contradiction draw does not match the blank sheet")
    return {"seed": ROUND2_SEED, "rebuilt_by": "evaluation/human_review/rebuild_arm_keys.py",
            "items": [{"item_id": row["item_id"], "arm": arm, "rank": r["rank"], "pair": r["pair"],
                       "verdict": r["verdict"]} for (r, arm), row in zip(picked, rows)]}


def rebuilt_keys():
    """Every key that can be rebuilt, by sheet kind (the names human_ratings.py uses)."""
    hypothesis, contradiction = round1_keys()
    return {"hypothesis": hypothesis, "contradiction": contradiction,
            "contradiction_round2": round2_contradiction_key()}


def main() -> int:
    for kind, key in rebuilt_keys().items():
        path = HERE / f"{kind}_key_DO_NOT_SHARE_WITH_ANNOTATORS.json"
        if path.exists():
            print(f"{path.name}: exists, left unchanged")
            continue
        path.write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
        print(f"{path.name}: rebuilt ({len(key['items'])} items)")
    print("hypothesis_round2 key: cannot be rebuilt from committed files (see module docstring)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
