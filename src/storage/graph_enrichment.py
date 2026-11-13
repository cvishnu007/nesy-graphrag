"""Additive, reversible enrichment of the existing Neo4j graph.

Two enrichments address measured weaknesses of the current graph:

1. Generic-concept flags. spaCy noun chunks include discourse phrases such as
   "this paper" (linked to 7,013 papers) that make every concept path two hops
   long and every paper pair look related. Each Concept node receives ``df``
   (number of linked papers) and ``generic`` (bool). Nothing is deleted; the
   reasoning queries skip ``generic = true`` concepts when
   ``CONCEPT_EXCLUDE_GENERIC`` is on.
2. Semantic kNN edges. 71% of papers have no in-corpus citation edge. Each paper
   receives ``SIMILAR_TO`` edges to its top-k nearest neighbours, computed from
   the embeddings already stored in Chroma (no re-encoding).

Undo:  MATCH ()-[r:SIMILAR_TO]->() DELETE r
       MATCH (e:SciEntity) DETACH DELETE e   (scientific entities)
       MATCH (c:Concept) REMOVE c.df, c.generic, c.generic_reason
"""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping

import numpy as np

# Discourse, deictic, and pronoun phrases that carry no scientific content.
GENERIC_PATTERNS = (
    r"^(this|these|that|those|our|their|its|such|each|both|other|various|several|many|most|some|all|any|further|previous|recent|existing|prior|current|future|new|novel|different|same|first|second|third|one|two|three)\b",
    r"^(them|they|it|we|us|what|which|who|whom|whose|itself|themselves)$",
    r"\b(paper|study|work|article|research|researchers|approach|approaches|method|methods|technique|techniques|results|experiments|literature|survey|review|authors|addition|terms|example|order|recent years|the-art|state-of-the-art)$",
)
_GENERIC_RE = [re.compile(pattern) for pattern in GENERIC_PATTERNS]


def generic_reason(name: str, df: int, *, max_df: int) -> str | None:
    """Return why a concept is generic, or None when it is retained."""
    text = " ".join(str(name).lower().split())
    if not text:
        return "empty"
    if any(pattern.search(text) for pattern in _GENERIC_RE):
        return "discourse_phrase"
    if df > max_df:
        return "document_frequency"
    return None


def classify_concepts(frequencies: Mapping[str, int], *, paper_count: int,
                      max_df_fraction: float = 0.01) -> dict[str, dict]:
    """Classify every concept from its document frequency and surface form."""
    if paper_count <= 0:
        raise ValueError("paper_count must be positive")
    if not 0 < max_df_fraction <= 1:
        raise ValueError("max_df_fraction must be in (0, 1]")
    max_df = max(1, int(paper_count * max_df_fraction))
    result = {}
    for name, df in frequencies.items():
        reason = generic_reason(name, int(df), max_df=max_df)
        result[name] = {"df": int(df), "generic": reason is not None, "reason": reason}
    return result


def top_k_neighbours(ids: list[str], vectors, *, k: int = 5, block: int = 2048,
                     min_similarity: float = 0.0, device: str | None = None) -> list[dict]:
    """Exact cosine kNN over all papers, excluding self matches."""
    if k <= 0:
        raise ValueError("k must be positive")
    if len(ids) != len(vectors):
        raise ValueError("ids and vectors must align")
    if len(set(ids)) != len(ids):
        raise ValueError("paper IDs must be unique")
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    matrix = torch.tensor(np.asarray(vectors, dtype=np.float32), device=device)
    matrix = torch.nn.functional.normalize(matrix, dim=1)
    take = min(k + 1, len(ids))
    edges = []
    for start in range(0, len(ids), block):
        scores = matrix[start:start + block] @ matrix.T
        rows = torch.arange(scores.shape[0], device=device)
        scores[rows, rows + start] = -2.0  # remove self
        values, indices = torch.topk(scores, take, dim=1)
        values, indices = values.cpu().numpy(), indices.cpu().numpy()
        for offset in range(scores.shape[0]):
            rank = 0
            for value, index in zip(values[offset], indices[offset]):
                if index == start + offset or value < min_similarity or rank >= k:
                    continue
                rank += 1
                edges.append({"source": ids[start + offset], "target": ids[index],
                              "score": round(float(value), 6), "rank": rank})
    return edges


def concept_frequencies(driver) -> tuple[dict[str, int], int]:
    with driver.session() as session:
        papers = session.run("MATCH (p:Paper) RETURN count(p) AS c").single()["c"]
        rows = session.run(
            "MATCH (c:Concept) OPTIONAL MATCH (c)<-[:RELATED_TO]-(p:Paper) "
            "RETURN c.name AS name, count(p) AS df"
        )
        return {row["name"]: row["df"] for row in rows}, papers


def write_concept_flags(driver, classified: Mapping[str, dict], *, batch: int = 5000) -> int:
    rows = [{"name": name, **value} for name, value in classified.items()]
    with driver.session() as session:
        for start in range(0, len(rows), batch):
            session.run(
                "UNWIND $rows AS row MATCH (c:Concept {name: row.name}) "
                "SET c.df = row.df, c.generic = row.generic, c.generic_reason = row.reason",
                rows=rows[start:start + batch],
            ).consume()
    return len(rows)


def write_similarity_edges(driver, edges: Iterable[dict], *, batch: int = 5000) -> int:
    rows = list(edges)
    with driver.session() as session:
        session.run("CREATE INDEX paper_id_lookup IF NOT EXISTS FOR (p:Paper) ON (p.id)").consume()
        for start in range(0, len(rows), batch):
            session.run(
                "UNWIND $rows AS row MATCH (a:Paper {id: row.source}) MATCH (b:Paper {id: row.target}) "
                "MERGE (a)-[r:SIMILAR_TO]->(b) SET r.score = row.score, r.rank = row.rank, "
                "r.source = 'specter_knn'",
                rows=rows[start:start + batch],
            ).consume()
    return len(rows)


def write_scientific_entities(driver, records, *, paper_count: int, max_df_fraction: float = 0.01,
                              max_per_paper: int = 15, batch: int = 2000) -> dict:
    """Write (:Paper)-[:MENTIONS]->(:SciEntity {name, type, df, generic}) additively."""
    from collections import Counter

    trimmed = [{"id": r["id"], "entities": r["entities"][:max_per_paper]} for r in records if r["entities"]]
    frequencies = Counter(e["text"] for r in trimmed for e in r["entities"])
    types = {}
    for record in trimmed:
        for entity in record["entities"]:
            types.setdefault(entity["text"], Counter())[entity["type"]] += 1
    classified = classify_concepts(frequencies, paper_count=paper_count, max_df_fraction=max_df_fraction)
    nodes = [{"name": name, "type": types[name].most_common(1)[0][0], **value} for name, value in classified.items()]
    with driver.session() as session:
        session.run("CREATE CONSTRAINT sci_entity_name IF NOT EXISTS FOR (e:SciEntity) REQUIRE e.name IS UNIQUE").consume()
        for start in range(0, len(nodes), batch):
            session.run("UNWIND $rows AS row MERGE (e:SciEntity {name: row.name}) "
                        "SET e.type = row.type, e.df = row.df, e.generic = row.generic, e.generic_reason = row.reason",
                        rows=nodes[start:start + batch]).consume()
        for start in range(0, len(trimmed), batch):
            session.run("UNWIND $rows AS row MATCH (p:Paper {id: row.id}) UNWIND row.entities AS entity "
                        "MATCH (e:SciEntity {name: entity.text}) MERGE (p)-[m:MENTIONS]->(e) SET m.type = entity.type",
                        rows=trimmed[start:start + batch]).consume()
    return {"papers_with_entities": len(trimmed), "entity_nodes": len(nodes),
            "mention_edges": sum(len(r["entities"]) for r in trimmed),
            "generic_nodes": sum(n["generic"] for n in nodes),
            "singleton_entities": sum(1 for n in nodes if n["df"] == 1),
            "by_type": dict(Counter(n["type"] for n in nodes)),
            "top_retained": sorted(((n["df"], n["name"], n["type"]) for n in nodes if not n["generic"]), reverse=True)[:30]}


def load_chroma_vectors(collection_name: str):
    import chromadb
    from src.utils.config import CHROMA_DIR

    collection = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(collection_name)
    ids = collection.get(include=[])["ids"]
    vectors = []
    for start in range(0, len(ids), 2000):
        batch = collection.get(ids=ids[start:start + 2000], include=["embeddings"])
        order = {pid: index for index, pid in enumerate(batch["ids"])}
        vectors.extend(batch["embeddings"][order[pid]] for pid in ids[start:start + 2000])
    return ids, vectors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["concepts", "similarity", "scientific-entities"])
    parser.add_argument("--input", default="data/s2_ner_scientific.json", help="scientific-entities input")
    parser.add_argument("--write", action="store_true", help="Apply additive writes; default is a dry run")
    parser.add_argument("--max-df-fraction", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument("--collection", help="Chroma collection whose vectors define similarity")
    parser.add_argument("--report", help="Optional JSON summary path")
    args = parser.parse_args(argv)
    from src.storage.neo4j_store import get_driver
    from src.utils.config import CHROMA_COLLECTION

    driver = get_driver()
    try:
        if args.task == "scientific-entities":
            payload = json.loads(open(args.input, encoding="utf-8").read())
            with driver.session() as session:
                papers = session.run("MATCH (p:Paper) RETURN count(p) AS c").single()["c"]
            if not args.write:
                raise SystemExit("scientific-entities requires --write (it has no dry-run summary)")
            summary = {"task": "scientific-entities", "model": payload.get("model"),
                       **write_scientific_entities(driver, payload["records"], paper_count=papers,
                                                   max_df_fraction=args.max_df_fraction)}
        elif args.task == "concepts":
            frequencies, papers = concept_frequencies(driver)
            classified = classify_concepts(frequencies, paper_count=papers, max_df_fraction=args.max_df_fraction)
            generic = [n for n, v in classified.items() if v["generic"]]
            edges_removed = sum(classified[n]["df"] for n in generic)
            summary = {
                "task": "concepts", "paper_count": papers, "concept_count": len(classified),
                "max_df": max(1, int(papers * args.max_df_fraction)),
                "generic_count": len(generic),
                "generic_by_reason": {r: sum(1 for v in classified.values() if v["reason"] == r)
                                      for r in ("discourse_phrase", "document_frequency")},
                "concept_edges_ignored": edges_removed,
                "singleton_concepts": sum(1 for v in classified.values() if v["df"] == 1),
                "top_generic": sorted(((classified[n]["df"], n) for n in generic), reverse=True)[:30],
                "top_retained": sorted(((v["df"], n) for n, v in classified.items() if not v["generic"]), reverse=True)[:30],
                "written": write_concept_flags(driver, classified) if args.write else 0,
            }
        else:
            collection = args.collection or CHROMA_COLLECTION
            ids, vectors = load_chroma_vectors(collection)
            edges = top_k_neighbours(ids, vectors, k=args.k, min_similarity=args.min_similarity)
            scores = sorted(edge["score"] for edge in edges)
            summary = {
                "task": "similarity", "collection": collection, "paper_count": len(ids), "k": args.k,
                "edge_count": len(edges),
                "score_quantiles": {q: scores[int(q * (len(scores) - 1))] for q in (0.0, 0.1, 0.5, 0.9, 1.0)} if scores else {},
                "written": write_similarity_edges(driver, edges) if args.write else 0,
            }
    finally:
        driver.close()
    text = json.dumps(summary, indent=2, default=str)
    if args.report:
        from pathlib import Path
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
