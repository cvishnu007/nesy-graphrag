"""Round 2: blinded test sheets for the revised hypothesis and contradiction methods.

Run from the repository root with the live stores up and the BGE configuration in .env:
    python evaluation/human_review/build_round2_kit.py

Hypotheses: the 14 stratified queries NOT used in round 1 (test set). Per query,
four arms: vector neighbour, original structural hole, problem-bridge (selected on
dev, margin 0.10) and problem-bridge restricted to specific entities (tied on
dev; reported as secondary). A paper chosen by several arms appears once, and
the key lists every arm that chose it.

Contradictions: 40 pairs from the citation-grounded index
(results/reasoning/contradiction_index_20261114): every pair the verdict model
called a contradiction, then unflagged pairs from the top 150 by NLI, then pairs
from ranks 151+, so precision can be estimated with and without the verdict filter.
"""
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SEED = 20261115
OUT = Path("evaluation/human_review")
QUERIES = "evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json"
INDEX = Path("results/reasoning/contradiction_index_20261114/index.jsonl")


def main():
    from build_review_kit import cited_by_any, pick_queries
    from src.pipeline.hypothesis import generate_hypotheses
    from src.pipeline.hypothesis_bridge import generate_bridged_hypotheses
    from src.pipeline.retrieval import neural_retrieve
    from src.storage.neo4j_store import get_driver
    from src.utils import config

    if (config.CONCEPT_SOURCE != "scientific" or config.EMBEDDING_MODEL != "BAAI/bge-base-en-v1.5"
            or not config.CONCEPT_EXCLUDE_GENERIC):
        raise SystemExit("Use the documented configuration: BGE, CONCEPT_SOURCE=scientific, "
                         "CONCEPT_EXCLUDE_GENERIC=true (round 1 ran without the generic filter; see RESULTS 4b)")
    corpus = {p["id"]: p for p in json.loads(Path(config.CLEAN_FILE).read_text(encoding="utf-8"))}
    used = {q["query_id"] for q in pick_queries()}
    test_queries = [q for q in json.loads(Path(QUERIES).read_text(encoding="utf-8"))["queries"]
                    if q["query_id"] not in used]
    rng = random.Random(SEED)
    hypothesis_items = []
    driver = get_driver()
    try:
        for query in test_queries:
            text = query["query"]
            neighbours = neural_retrieve(text, top_k=15)
            top_ids = [p["id"] for p in neighbours[:5]]
            context = " | ".join(corpus[i]["clean_title"] for i in top_ids[:3])
            chosen = {}
            old = generate_hypotheses(driver, text, top_k=1)
            if old:
                chosen.setdefault(old[0]["id"], []).append("graph_structural_hole")
            vector = next((p for p in neighbours[5:] if not cited_by_any(driver, top_ids, p["id"])), None)
            if vector:
                chosen.setdefault(vector["id"], []).append("vector_neighbour")
            for arm, options in (("bridge_m10", {"margin": 0.10}),
                                 ("bridge_m10_specific", {"margin": 0.10, "specific_only": True})):
                found = generate_bridged_hypotheses(driver, text, top_k=1, **options)
                if found:
                    chosen.setdefault(found[0]["id"], []).append(arm)
            for paper_id, arms in chosen.items():
                paper = corpus[paper_id]
                hypothesis_items.append({
                    "query": text, "query_papers": context, "candidate_title": paper["clean_title"],
                    "candidate_year": paper.get("year", ""), "candidate_abstract": paper["clean_abstract"][:1500],
                    "_key": {"query_id": query["query_id"], "arms": arms, "paper_id": paper_id,
                             "query_paper_ids": top_ids}})
            print(f"{query['query_id']}: {len(chosen)} distinct candidates", flush=True)
    finally:
        driver.close()

    index = [json.loads(line) for line in INDEX.read_text(encoding="utf-8").splitlines()]
    flagged = [r for r in index if r["verdict"] == "CONTRADICTION"]
    top_other = [r for r in index if r["rank"] <= 150 and r["verdict"] != "CONTRADICTION"]
    lower = [r for r in index if r["rank"] > 150]
    remaining = 40 - len(flagged)
    picked = ([(r, "index_verdict_contradiction") for r in flagged]
              + [(r, "index_top150_unflagged") for r in rng.sample(top_other, min(len(top_other), remaining // 2))]
              + [(r, "index_rank151_plus") for r in rng.sample(lower, min(len(lower), remaining - remaining // 2))])
    contradiction_items = []
    for row, arm in picked:
        first, second = corpus[row["citing_id"]], corpus[row["cited_id"]]
        contradiction_items.append({
            "query": "(corpus-wide index)",
            "paper_1_title": first["clean_title"], "paper_1_year": first.get("year", ""),
            "finding_1": row["evidence_1"]["text"],
            "paper_2_title": second["clean_title"], "paper_2_year": second.get("year", ""),
            "finding_2": row["evidence_2"]["text"],
            "_key": {"arm": arm, "rank": row["rank"], "pair": row["pair"], "verdict": row["verdict"],
                     "nli_contradiction": row["nli_contradiction"], "shared_subjects": row["shared_subjects"]}})

    for items, prefix, fields in ((hypothesis_items, "hypothesis_round2", ["relevance_1_5", "novelty_1_5", "promise_1_5", "notes"]),
                                  (contradiction_items, "contradiction_round2", ["label_0_1_2", "notes"])):
        rng.shuffle(items)
        rows, key = [], []
        letter = "HT" if prefix.startswith("hyp") else "CT"
        for number, item in enumerate(items, 1):
            item_id = f"{letter}{number:03d}"
            hidden = item.pop("_key")
            rows.append({"item_id": item_id, **item, **{f: "" for f in fields}})
            key.append({"item_id": item_id, **hidden})
        with open(OUT / f"{prefix}_sheet_blank.csv", "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (OUT / f"{prefix}_key_DO_NOT_SHARE_WITH_ANNOTATORS.json").write_text(json.dumps(
            {"seed": SEED, "design": __doc__, "concept_source": config.CONCEPT_SOURCE,
             "concept_exclude_generic": config.CONCEPT_EXCLUDE_GENERIC, "embedding_model": config.EMBEDDING_MODEL,
             "items": key}, indent=1) + "\n")
        print(f"{prefix}: {len(rows)} items", flush=True)


if __name__ == "__main__":
    main()
