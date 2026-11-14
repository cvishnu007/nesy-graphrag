"""Problem-bridge, novel-method hypothesis candidates (revision of the structural-hole generator).

Human ratings of 30 queries (evaluation/human_review) found the original
structural-hole candidates *less relevant* than a nearest-neighbour baseline
(mean 3.27 vs 4.20 of 5, p = 0.002) and no more novel. Rater notes point at the
cause: candidates linked by broad shared concepts ("dl models") are off-topic.
A first fix (link through specific query-topic entities) produced relevant but
obvious same-topic papers. This generator follows the literature-based-discovery
pattern instead ("apply method M to problem T"):

1. **Shared problem.** The candidate shares a specific Task or Material entity
   with the query papers (non-generic, IDF-weighted, so broad terms count little).
2. **Novel method.** The candidate mentions Method entities that none of the
   query's ``exclude_rank`` nearest papers mention. These are the proposed bridge.
3. **Relevance floor.** Query similarity >= the weakest seed paper's similarity minus ``margin``.
4. **Non-obvious.** Outside the vector top ``exclude_rank`` and not citation-linked to the seeds.

Score = IDF(shared problems) x (1 + IDF(novel methods) / 10) x query similarity.
"""
from __future__ import annotations

import math

PROBLEM_TYPES = ("Task", "Material")


def is_specific(name):
    """Multi-word names ("graph contrastive learning") or acronyms ("mpnn"), not lone common words."""
    words = name.split()
    return len(words) >= 2 or "-" in name or (len(name) <= 6 and name.isalnum() and not name.isalpha()) or name.isupper()         or (len(words) == 1 and len(name) <= 5)


def idf(dfs, paper_count):
    return sum(math.log(paper_count / max(1, df)) for df in dfs)


def bridge_score(problem_dfs, method_dfs, similarity, paper_count):
    return idf(problem_dfs, paper_count) * (1 + idf(method_dfs, paper_count) / 10) * similarity


def generate_bridged_hypotheses(driver, query, top_k=5, *, seeds=5, exclude_rank=15, margin=0.08,
                                max_df=500, pool=400, paper_count=47619, require_novel_method=True,
                                specific_only=False):
    from src.pipeline.retrieval import neural_retrieve
    from src.storage.chroma_store import score_papers_against_query

    neighbours = neural_retrieve(query, top_k=exclude_rank)
    if not neighbours:
        return []
    seed_papers = neighbours[:seeds]
    seed_ids = [p["id"] for p in seed_papers]
    neighbour_ids = [p["id"] for p in neighbours]
    floor = min(p["score"] for p in seed_papers) - margin
    cypher = """
        UNWIND $ids AS pid
        MATCH (q:Paper {id: pid})-[:MENTIONS]->(e:SciEntity)<-[:MENTIONS]-(c:Paper)
        WHERE NOT c.id IN $exclude AND NOT coalesce(e.generic, false)
          AND e.type IN $types AND e.df >= 2 AND e.df <= $max_df
        WITH c, collect(DISTINCT {name: e.name, df: e.df}) AS problems, collect(DISTINCT q.id) AS support
        WHERE NOT EXISTS { MATCH (s:Paper)-[:CITES]-(c) WHERE s.id IN $ids }
        CALL (c) {
            MATCH (c)-[:MENTIONS]->(m:SciEntity {type: 'Method'})
            WHERE NOT coalesce(m.generic, false) AND m.df >= 2
              AND NOT EXISTS { MATCH (n:Paper)-[:MENTIONS]->(m) WHERE n.id IN $exclude }
            RETURN collect(DISTINCT {name: m.name, df: m.df}) AS methods
        }
        RETURN c.id AS id, c.title AS title, c.year AS year, problems, methods, support
        ORDER BY size(problems) DESC, size(support) DESC
        LIMIT $pool
    """
    with driver.session() as session:
        records = list(session.run(cypher, ids=seed_ids, exclude=neighbour_ids, types=list(PROBLEM_TYPES),
                                   max_df=max_df, pool=pool))
    similarity = score_papers_against_query(query, [r["id"] for r in records])
    candidates = []
    for record in records:
        sim = similarity.get(record["id"])
        if sim is None or sim < floor or (require_novel_method and not record["methods"]):
            continue
        problems = sorted((e for e in record["problems"] if not specific_only or is_specific(e["name"])),
                          key=lambda e: e["df"])
        methods = sorted((e for e in record["methods"] if not specific_only or is_specific(e["name"])),
                         key=lambda e: e["df"])
        if not problems or (require_novel_method and not methods):
            continue
        candidates.append({
            "id": record["id"], "title": record["title"], "year": record["year"],
            "query_similarity": round(sim, 4), "relevance_floor": round(floor, 4),
            "shared_problems": [e["name"] for e in problems], "novel_methods": [e["name"] for e in methods],
            "supporting_paper_ids": record["support"],
            "evidence_score": round(bridge_score([e["df"] for e in problems], [e["df"] for e in methods], sim,
                                                 paper_count), 4),
            "candidate_method": "problem_bridge_novel_method",
        })
    return sorted(candidates, key=lambda c: (-c["evidence_score"], c["id"]))[:top_k]


GENERIC_HEADS = {"approach", "approaches", "method", "methods", "technique", "techniques", "algorithm", "algorithms",
                 "measure", "measures", "model", "models", "strategy", "strategies", "system", "systems",
                 "framework", "frameworks", "solution", "solutions", "tool", "tools", "mechanism", "mechanisms"}


def is_informative(name, kind):
    """Readable hypothesis terms: no generic head with one modifier; no ambiguous short task acronyms."""
    words = name.split()
    if len(words) <= 2 and words[-1] in GENERIC_HEADS:
        return False
    if kind == "Task" and len(name) <= 3:
        return False
    return True
