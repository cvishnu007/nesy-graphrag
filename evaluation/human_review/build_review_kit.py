"""Build the blinded hypothesis and contradiction rating sheets plus hidden keys.

Run from the repository root with the live stores up:
    CONCEPT_SOURCE=scientific EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 CHROMA_COLLECTION=s2_papers_bge     EMBEDDING_QUERY_PREFIX="Represent this sentence for searching relevant passages: " GRAPH_GATE_MODE=relative     GRAPH_RELATIVE_MARGIN=0.03 GRAPH_EXPANSION_EDGES="CITES|SIMILAR_TO" GRAPH_CANDIDATE_LIMIT=50     BM25_FUSION_WEIGHT=1.0 python evaluation/human_review/build_review_kit.py

Hypothesis sheet: for each query, one structural-hole candidate from the concept
graph (``generate_hypotheses``) and one nearest-neighbour baseline (vector rank
6-15, not cited by the top 5). Both arms show the same fields, so raters cannot
tell them apart. Rows are shuffled. The key records which arm each row came from.

Contradiction sheet: for each query, the top 2 NLI-ranked contradiction
candidates among the top 20 retrieved papers, plus a lower-ranked control pair (rank 5-15)
for half of the queries, to estimate how much the ranking adds.

Keys are gitignored (``*DO_NOT_SHARE*``) because annotators have repository access.
"""
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SEED = 20261114
QUERIES = "evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json"
OUT = Path("evaluation/human_review")
N_QUERIES = 30


def pick_queries():
    queries = json.loads(Path(QUERIES).read_text(encoding="utf-8"))["queries"]
    ranked = sorted(queries, key=lambda q: hashlib.sha256(f"{SEED}|{q['query_id']}".encode()).hexdigest())
    return sorted(ranked[:N_QUERIES], key=lambda q: q["query_id"])


def shared_concepts(driver, query_ids, candidate_id, limit=6):
    with driver.session() as session:
        record = session.run(
            "MATCH (q:Paper)-[:MENTIONS]->(e:SciEntity)<-[:MENTIONS]-(c:Paper {id: $cid}) "
            "WHERE q.id IN $ids AND NOT coalesce(e.generic, false) "
            "WITH e, count(DISTINCT q) AS support ORDER BY support DESC, e.df ASC "
            "RETURN collect(e.name)[..$limit] AS names", ids=query_ids, cid=candidate_id, limit=limit).single()
    return record["names"] if record else []


def cited_by_any(driver, query_ids, candidate_id):
    with driver.session() as session:
        return session.run("MATCH (q:Paper)-[:CITES]-(c:Paper {id: $cid}) WHERE q.id IN $ids RETURN count(*) AS n",
                           ids=query_ids, cid=candidate_id).single()["n"] > 0


def write_sheet(name, rows, fields):
    with open(OUT / name, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    from src.pipeline.contradiction_nli import detect_contradictions_nli
    from src.pipeline.hypothesis import generate_hypotheses
    from src.pipeline.retrieval import neural_retrieve
    from src.storage.neo4j_store import get_driver
    from src.utils import config

    if config.CONCEPT_SOURCE != "scientific" or config.EMBEDDING_MODEL != "BAAI/bge-base-en-v1.5":
        raise SystemExit("Use the selected configuration: CONCEPT_SOURCE=scientific and the BGE settings (see docstring)")
    corpus = {p["id"]: p for p in json.loads(Path(config.CLEAN_FILE).read_text(encoding="utf-8"))}
    rng = random.Random(SEED)
    driver = get_driver()
    hypothesis_items, contradiction_items = [], []
    try:
        for index, query in enumerate(pick_queries()):
            text = query["query"]
            neighbours = neural_retrieve(text, top_k=15)
            top_ids = [p["id"] for p in neighbours[:5]]
            context = " | ".join(corpus[i]["clean_title"] for i in top_ids[:3])
            graph = generate_hypotheses(driver, text, top_k=1)
            vector = next((p for p in neighbours[5:] if not cited_by_any(driver, top_ids, p["id"])
                           and (not graph or p["id"] != graph[0]["id"])), None)
            for arm, candidate_id in (("graph_structural_hole", graph[0]["id"] if graph else None),
                                      ("vector_neighbour", vector["id"] if vector else None)):
                if candidate_id:
                    paper = corpus[candidate_id]
                    hypothesis_items.append({
                        "query": text, "query_papers": context, "candidate_title": paper["clean_title"],
                        "candidate_year": paper.get("year", ""), "candidate_abstract": paper["clean_abstract"][:1500],
                        "_key": {"query_id": query["query_id"], "arm": arm, "paper_id": candidate_id,
                                 # kept out of the sheet: graph candidates have ~2x more, which unblinds the arms
                                 "linking_concepts": "; ".join(shared_concepts(driver, top_ids, candidate_id)),
                                 "query_paper_ids": top_ids}})
            ranked = detect_contradictions_nli(driver, text, top_k=15, retrieve_k=20)
            chosen = [(rank, "top_ranked") for rank in range(min(2, len(ranked)))]
            if index % 2 == 0 and len(ranked) >= 5:
                chosen.append((rng.randrange(4, len(ranked)), "lower_ranked_control"))
            for rank, arm in chosen:
                pair = ranked[rank]
                contradiction_items.append({
                    "query": text,
                    "paper_1_title": pair["paper1"]["title"], "paper_1_year": pair["paper1"].get("year", ""),
                    "finding_1": pair["evidence_1"]["text"],
                    "paper_2_title": pair["paper2"]["title"], "paper_2_year": pair["paper2"].get("year", ""),
                    "finding_2": pair["evidence_2"]["text"],
                    "_key": {"query_id": query["query_id"], "arm": arm, "rank": rank + 1,
                             "paper_1_id": pair["paper1"]["id"], "paper_2_id": pair["paper2"]["id"],
                             "evidence_1_id": pair["evidence_1"].get("id"), "evidence_2_id": pair["evidence_2"].get("id"),
                             "nli_contradiction": pair["nli_contradiction"]}})
            print(f"{query['query_id']}: hypothesis arms {len(graph)}+{int(vector is not None)}, "
                  f"contradiction pairs {len(chosen)}", flush=True)
    finally:
        driver.close()

    for items, prefix, rating_fields in (
        (hypothesis_items, "hypothesis", ["relevance_1_5", "novelty_1_5", "promise_1_5", "notes"]),
        (contradiction_items, "contradiction", ["label_0_1_2", "notes"]),
    ):
        rng.shuffle(items)
        rows, key = [], []
        letter = prefix[0].upper()
        for number, item in enumerate(items, 1):
            item_id = f"{letter}{number:03d}"
            hidden = item.pop("_key")
            rows.append({"item_id": item_id, **item, **{field: "" for field in rating_fields}})
            key.append({"item_id": item_id, **hidden})
        write_sheet(f"{prefix}_sheet_blank.csv", rows, list(rows[0]))
        (OUT / f"{prefix}_key_DO_NOT_SHARE_WITH_ANNOTATORS.json").write_text(json.dumps(
            {"seed": SEED, "queries": QUERIES, "concept_source": config.CONCEPT_SOURCE,
             "embedding_model": config.EMBEDDING_MODEL, "items": key}, indent=1) + "\n")
        print(f"{prefix}: {len(rows)} items", flush=True)


if __name__ == "__main__":
    main()
