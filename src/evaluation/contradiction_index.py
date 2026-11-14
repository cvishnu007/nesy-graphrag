"""Build the corpus-wide, citation-grounded contradiction index and screen it with a verdict model.

Citations: original corpus-internal references plus ``cites_added`` from the
densification plan (Semantic Scholar supplement + OpenAlex). Candidates come from
``contradiction_citation.build_corpus_index``. The top ``--verdict-top`` pairs
by NLI score get a structured verdict from the local model (the filter measured
on 68 human-labelled pairs: 4.8% false alarms). Output:
``results/reasoning/contradiction_index_<date>/index.jsonl`` and ``summary.json``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

PLAN = Path("data/citation_densify_plan.json")


def citation_map(corpus, plan):
    from src.ingestion.citation_densify import parse_refs

    ids = {p["id"] for p in corpus}
    cites = defaultdict(set)
    for paper in corpus:
        for ref in parse_refs(paper.get("references")):
            if ref in ids and ref != paper["id"]:
                cites[paper["id"]].add(ref)
    for source, target in plan["cites_added"]:
        cites[source].add(target)
    return cites


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="results/reasoning/contradiction_index_20261114")
    parser.add_argument("--verdict-top", type=int, default=150)
    parser.add_argument("--llm", default="ollama:qwen2.5:7b-instruct")
    args = parser.parse_args(argv)
    from src.evaluation.judge_agreement import make_client
    from src.evaluation.contradiction_human_eval import verdict
    from src.pipeline.contradiction_citation import build_corpus_index
    from src.storage.neo4j_store import get_driver
    from src.utils.config import CLEAN_FILE

    corpus = json.loads(Path(CLEAN_FILE).read_text(encoding="utf-8"))
    cites = citation_map(corpus, json.loads(PLAN.read_text()))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    driver = get_driver()
    try:
        index, stats = build_corpus_index(driver, corpus, cites)
    finally:
        driver.close()
    print(json.dumps(stats), f"candidates={len(index)}", flush=True)

    checkpoint = output / "verdicts.jsonl"
    done = {}
    if checkpoint.exists():
        done = {r["pair"]: r for r in map(json.loads, checkpoint.read_text(encoding="utf-8").splitlines())
                if not r["raw"].startswith("ERROR")}
    client = make_client(args.llm)
    with checkpoint.open("a", encoding="utf-8") as stream:
        for candidate in index[:args.verdict_top]:
            pair = f"{candidate['paper1']['id']}|{candidate['paper2']['id']}"
            if pair in done:
                continue
            try:
                raw, parsed = verdict(client, {"papers": (candidate["paper1"], candidate["paper2"]),
                                               "evidence": (candidate["evidence_1"], candidate["evidence_2"])})
            except Exception as error:  # retried on resume, never scored
                raw, parsed = f"ERROR {type(error).__name__}", {"verdict": "UNKNOWN", "confidence": None, "valid": False}
            done[pair] = {"pair": pair, "verdict": parsed["verdict"], "confidence": parsed["confidence"],
                          "valid": parsed["valid"], "raw": raw[:2000]}
            stream.write(json.dumps(done[pair]) + "\n")
            stream.flush()
    with (output / "index.jsonl").open("w", encoding="utf-8") as stream:
        for rank, candidate in enumerate(index, 1):
            pair = f"{candidate['paper1']['id']}|{candidate['paper2']['id']}"
            row = {"rank": rank, "pair": pair, "citing_id": candidate["paper1"]["id"],
                   "cited_id": candidate["paper2"]["id"], "nli_contradiction": candidate["nli_contradiction"],
                   "shared_subjects": candidate["shared_subjects"],
                   "evidence_1": candidate["evidence_1"], "evidence_2": candidate["evidence_2"],
                   "verdict": done.get(pair, {}).get("verdict")}
            stream.write(json.dumps(row) + "\n")
    verdicts = Counter(done[p]["verdict"] for p in done)
    summary = {**stats, "candidates": len(index), "verdict_model": args.llm, "verdicted": len(done),
               "verdict_counts": dict(verdicts),
               "nli_ge_0_5": sum(c["nli_contradiction"] >= 0.5 for c in index)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
