"""Related-work suggestions: papers a given paper (or topic) should cite or discuss.

Uses the configuration evaluated in docs/RESULTS.md §2f (citation recommendation,
+0.068 Recall@20 over BGE + BM25 on 400 test papers): reciprocal-rank fusion of
BGE neighbours, BM25 and a co-citation graph signal. The graph signal is the set
of papers cited (``CITES`` or ``CITES_ADDED``) by the input's 50 nearest BGE
neighbours, weighted by neighbour similarity.

Differences from the evaluated setting: the input is encoded as a query (with the
BGE query prefix), not as a stored document vector, and there is no publication
year filter unless ``max_year`` is given. With ``max_year``, neighbours, BM25
results and co-citation candidates are all filtered by year before fusion, so the
fused list is still filled to ``top_k`` (papers with no recorded year are kept).
"""
from __future__ import annotations

from collections import defaultdict

NEIGHBOURS = 50
GRAPH_WEIGHT = 1.0
RRF_K = 60
YEAR_FILTER_POOL = 4  # with max_year, fetch this many times more candidates before filtering


def within_year(year, max_year):
    """True if no limit is set, the year is unknown, or it is no later than ``max_year``."""
    if max_year is None or year in (None, ""):
        return True
    try:
        return int(year) <= int(max_year)
    except (TypeError, ValueError):
        return True


def fuse(bge, lexical, graph, *, top_k, exclude=(), graph_weight=GRAPH_WEIGHT):
    """Weighted RRF over three ranked id lists; returns ids with per-source ranks."""
    scores, ranks = defaultdict(float), defaultdict(dict)
    for source, ranking, weight in (("bge", bge, 1.0), ("bm25", lexical, 1.0), ("cocitation", graph, graph_weight)):
        for rank, paper_id in enumerate(ranking, 1):
            if paper_id in exclude:
                continue
            scores[paper_id] += weight / (RRF_K + rank)
            ranks[paper_id][source] = rank
    ordered = sorted(scores, key=lambda pid: (-scores[pid], pid))[:top_k]
    return [(pid, round(scores[pid], 5), ranks[pid]) for pid in ordered]


def cocitation(driver, neighbours, *, exclude=(), max_year=None):
    """Papers cited by the neighbours, scored by summed neighbour similarity."""
    similarity = {p["id"]: float(p.get("score") or 0.0) for p in neighbours}
    with driver.session() as session:
        records = list(session.run(
            "UNWIND $ids AS nid MATCH (n:Paper {id: nid})-[:CITES|CITES_ADDED]->(c:Paper) "
            "RETURN c.id AS id, c.year AS year, collect(DISTINCT nid) AS citing", ids=list(similarity)))
    scored = {r["id"]: (sum(similarity[n] for n in r["citing"]), list(r["citing"]))
              for r in records if r["id"] not in exclude and within_year(r.get("year"), max_year)}
    ranking = sorted(scored, key=lambda pid: (-scored[pid][0], pid))
    return ranking, {pid: citing for pid, (_, citing) in scored.items()}


def citation_graph(driver, results, neighbours, citers, *, max_neighbours=12):
    """Nodes and edges for drawing the suggestions with their citation evidence.

    Edges: a similar paper (neighbour) citing a suggestion, which is the co-citation
    evidence, and citations among the suggestions themselves. Only the neighbours
    citing the most suggestions are kept so the drawing stays readable.
    """
    suggestion_ids = [r["id"] for r in results]
    chosen = set(suggestion_ids)
    support = defaultdict(int)
    for pid in suggestion_ids:
        for nid in citers.get(pid, ()):
            if nid not in chosen:
                support[nid] += 1
    keep = sorted(support, key=lambda nid: (-support[nid], nid))[:max_neighbours]
    by_id = {p["id"]: p for p in neighbours}
    edges = [{"source": nid, "target": pid, "kind": "cocitation"}
             for pid in suggestion_ids for nid in citers.get(pid, ()) if nid in keep]
    with driver.session() as session:
        edges += [{"source": r["a"], "target": r["b"], "kind": "cites"} for r in session.run(
            "UNWIND $ids AS a MATCH (x:Paper {id: a})-[:CITES|CITES_ADDED]->(y:Paper) "
            "WHERE y.id IN $ids AND y.id <> a RETURN DISTINCT a, y.id AS b", ids=suggestion_ids)]
    nodes = [{"id": nid, "title": by_id[nid].get("title", ""), "year": by_id[nid].get("year"),
              "role": "neighbour", "cites_suggestions": support[nid]} for nid in keep if nid in by_id]
    return {"nodes": nodes, "edges": edges}


def related_work(driver, text, top_k=10, *, paper_id=None, max_year=None):
    from src.pipeline.retrieval import lexical_retrieve
    from src.storage.chroma_store import query as chroma_query

    exclude = set()
    if paper_id:
        exclude.add(paper_id)
        with driver.session() as session:
            exclude |= {r["id"] for r in session.run(
                "MATCH (c:Paper)-[:CITES|CITES_ADDED]->(:Paper {id: $pid}) RETURN c.id AS id", pid=paper_id)}
    pool = 1 if max_year is None else YEAR_FILTER_POOL
    neighbours = [p for p in chroma_query(text, top_k=NEIGHBOURS * pool + len(exclude))
                  if p["id"] not in exclude and within_year(p.get("year"), max_year)][:NEIGHBOURS]
    graph, citing_counts = cocitation(driver, neighbours, exclude=exclude, max_year=max_year)
    lexical = [p["id"] for p in lexical_retrieve(text, 100 * pool) if within_year(p.get("year"), max_year)][:100]
    fused = fuse([p["id"] for p in neighbours], lexical, graph[:100], top_k=top_k, exclude=exclude)
    with driver.session() as session:
        meta = {r["id"]: r for r in session.run(
            "UNWIND $ids AS pid MATCH (p:Paper {id: pid}) RETURN p.id AS id, p.title AS title, p.year AS year, "
            "p.abstract AS abstract", ids=[pid for pid, _, _ in fused])}
    results = []
    for pid, score, ranks in fused:
        record = meta.get(pid)
        if record is None or not within_year(record["year"], max_year):
            continue
        reasons = []
        if "cocitation" in ranks:
            reasons.append(f"cited by {len(citing_counts.get(pid, ()))} of the {len(neighbours)} most similar papers")
        if "bge" in ranks:
            reasons.append(f"semantic match #{ranks['bge']}")
        if "bm25" in ranks:
            reasons.append(f"keyword match #{ranks['bm25']}")
        results.append({"id": pid, "title": record["title"], "year": record["year"],
                        "abstract": record["abstract"], "score": score, "source_ranks": ranks,
                        "cited_by_neighbours": len(citing_counts.get(pid, ())), "why": "; ".join(reasons)})
    graph = citation_graph(driver, results, neighbours, citing_counts)
    attach_citation_evidence(results, neighbours, citing_counts, graph)
    return {"query": text, "neighbours": len(neighbours), "papers": results, "graph": graph}


def attach_citation_evidence(results, neighbours, citers, graph):
    """Add, per suggestion, the titled papers behind its citation signal.

    ``cited_by_similar``: every one of the similar papers whose reference list includes
    the suggestion (the co-citation evidence), most similar first.
    ``cites`` / ``cited_by``: citations between the suggestions themselves.
    """
    rank = {p["id"]: i for i, p in enumerate(neighbours)}
    info = {p["id"]: {"id": p["id"], "title": p.get("title", ""), "year": p.get("year")} for p in neighbours}
    by_id = {r["id"]: {"id": r["id"], "title": r["title"], "year": r["year"]} for r in results}
    inner = [(e["source"], e["target"]) for e in graph.get("edges", []) if e["kind"] == "cites"]
    for r in results:
        similar = sorted((n for n in citers.get(r["id"], ()) if n in info), key=lambda n: rank[n])
        r["cited_by_similar"] = [info[n] for n in similar]
        r["cites"] = [by_id[b] for a, b in inner if a == r["id"] and b in by_id]
        r["cited_by"] = [by_id[a] for a, b in inner if b == r["id"] and a in by_id]
