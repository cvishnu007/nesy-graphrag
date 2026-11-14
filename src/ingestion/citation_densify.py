"""Densify the citation graph without changing the base corpus or its CITES edges.

Only 10,899 of 47,619 papers came with reference lists (Semantic Scholar hides
most lists for publisher-licensing reasons), and only 3.6% of references resolve
inside the corpus, so 71% of papers have no citation edge. This module adds
three opt-in layers. Each uses its own label or relationship type, so the
original graph, retrieval defaults, and frozen evaluations are unchanged until a
variant enables them (``GRAPH_EXPANSION_EDGES``):

* ``CITES_ADDED``: corpus-internal citations from reference lists fetched later
  (``data/s2_references_supplement.json``);
* ``COUPLED_WITH``: bibliographic coupling between corpus papers that share
  references (Kessler, 1963), cosine-normalised, top-k per paper;
* ``(:ExternalPaper)`` with ``CITES_EXTERNAL``: the papers outside the corpus
  most cited by it (citation closure), for context and hypothesis evidence.
  They never carry the ``Paper`` label, so retrieval cannot return them.

Stages: ``fetch-s2`` -> ``build`` -> ``fetch-external`` -> ``write --write`` -> ``stats``.
Undo: delete ``CITES_ADDED``/``COUPLED_WITH``/``CITES_EXTERNAL`` relationships and
``ExternalPaper`` nodes; base data is never touched.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SUPPLEMENT = Path("data/s2_references_supplement.json")
EXTERNAL = Path("data/s2_external_papers.json")
PLAN = Path("data/citation_densify_plan.json")


def parse_refs(value):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    return [ref for ref in (value or []) if ref]


def load_corpus(path=None):
    from src.utils.config import CLEAN_FILE

    return json.loads(Path(path or CLEAN_FILE).read_text(encoding="utf-8"))


def reference_sets(corpus, supplement):
    refs = {p["id"]: set(parse_refs(p.get("references"))) for p in corpus}
    added = {}
    for paper_id, extra in supplement.items():
        if paper_id in refs:
            new = set(extra) - refs[paper_id]
            if new:
                added[paper_id] = new
                refs[paper_id] |= new
    return refs, added


def coupling_edges(refs, *, top_k=10, min_shared=2, max_ref_df=1000):
    """Cosine bibliographic coupling; references cited by > max_ref_df papers are skipped."""
    citing = defaultdict(list)
    for paper_id, values in refs.items():
        for ref in values:
            citing[ref].append(paper_id)
    shared = defaultdict(Counter)
    for papers in citing.values():
        if len(papers) < 2 or len(papers) > max_ref_df:
            continue
        for i, a in enumerate(papers):
            for b in papers[i + 1:]:
                shared[a][b] += 1
                shared[b][a] += 1
    best = {}
    for a, counts in shared.items():
        scored = sorted(((n / math.sqrt(len(refs[a]) * len(refs[b])), b, n) for b, n in counts.items()
                         if n >= min_shared), reverse=True)[:top_k]
        for score, b, n in scored:
            key = (a, b) if a < b else (b, a)
            if key not in best or best[key]["score"] < score:
                best[key] = {"source": key[0], "target": key[1], "shared": n, "score": round(score, 4)}
    return list(best.values())


def closure_targets(refs, corpus_ids, *, min_citing=10):
    counts = Counter(ref for values in refs.values() for ref in values if ref not in corpus_ids)
    return {ref: n for ref, n in counts.items() if n >= min_citing}


def _fetch(ids, fields, checkpoint: Path, *, keep, size=100):
    from src.ingestion.semantic_scholar_fetcher import SemanticScholarClient

    done = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    attempted = set(done.get("_attempted", []))
    todo = [i for i in ids if i not in attempted]
    client = SemanticScholarClient()
    for start in range(0, len(todo), size):
        batch = todo[start:start + size]
        result = client.request("POST", "/paper/batch", params={"fields": fields}, json_body={"ids": batch})
        for paper in result:
            if paper and paper.get("paperId"):
                value = keep(paper)
                if value:
                    done[paper["paperId"]] = value
        attempted.update(batch)
        done["_attempted"] = sorted(attempted)
        checkpoint.write_text(json.dumps(done))
        print(f"fetched {min(start + size, len(todo))}/{len(todo)} (kept {len(done) - 1})", flush=True)
    return {k: v for k, v in done.items() if k != "_attempted"}


def write_graph(driver, plan, external, *, batch=5000):
    with driver.session() as session:
        session.run("CREATE INDEX paper_id_lookup IF NOT EXISTS FOR (p:Paper) ON (p.id)").consume()
        session.run("CREATE CONSTRAINT external_paper_id IF NOT EXISTS FOR (e:ExternalPaper) REQUIRE e.id IS UNIQUE").consume()
        added = [{"source": s, "target": t} for s, t in plan["cites_added"]]
        for start in range(0, len(added), batch):
            session.run("UNWIND $rows AS row MATCH (a:Paper {id: row.source}) MATCH (b:Paper {id: row.target}) "
                        "MERGE (a)-[r:CITES_ADDED]->(b) SET r.source = 's2_reference_supplement'",
                        rows=added[start:start + batch]).consume()
        coupled = plan["coupled_with"]
        for start in range(0, len(coupled), batch):
            session.run("UNWIND $rows AS row MATCH (a:Paper {id: row.source}) MATCH (b:Paper {id: row.target}) "
                        "MERGE (a)-[r:COUPLED_WITH]->(b) SET r.shared = row.shared, r.score = row.score, "
                        "r.source = 'bibliographic_coupling'", rows=coupled[start:start + batch]).consume()
        nodes = [{"id": pid, **meta, "corpus_citations": plan["closure"][pid]}
                 for pid, meta in external.items() if pid in plan["closure"]]
        for start in range(0, len(nodes), batch):
            session.run("UNWIND $rows AS row MERGE (e:ExternalPaper {id: row.id}) "
                        "SET e.title = row.title, e.abstract = row.abstract, e.year = row.year, "
                        "e.venue = row.venue, e.citation_count = row.citationCount, "
                        "e.corpus_citations = row.corpus_citations", rows=nodes[start:start + batch]).consume()
        links = [{"source": s, "target": t} for s, t in plan["cites_external"] if t in external]
        for start in range(0, len(links), batch):
            session.run("UNWIND $rows AS row MATCH (a:Paper {id: row.source}) MATCH (e:ExternalPaper {id: row.target}) "
                        "MERGE (a)-[:CITES_EXTERNAL]->(e)", rows=links[start:start + batch]).consume()
    return {"cites_added": len(added), "coupled_with": len(coupled), "external_nodes": len(nodes),
            "cites_external": len(links)}


def graph_stats(driver):
    queries = {
        "papers": "MATCH (p:Paper) RETURN count(p) AS n",
        "isolated_original_cites": "MATCH (p:Paper) WHERE NOT (p)-[:CITES]-(:Paper) RETURN count(p) AS n",
        "isolated_cites_or_added": "MATCH (p:Paper) WHERE NOT (p)-[:CITES|CITES_ADDED]-(:Paper) RETURN count(p) AS n",
        "isolated_citation_layers": "MATCH (p:Paper) WHERE NOT (p)-[:CITES|CITES_ADDED|COUPLED_WITH]-(:Paper) "
                                    "RETURN count(p) AS n",
        "isolated_citation_layers_or_external": "MATCH (p:Paper) WHERE NOT (p)-[:CITES|CITES_ADDED|COUPLED_WITH]-(:Paper) "
                                                "AND NOT (p)-[:CITES_EXTERNAL]->() RETURN count(p) AS n",
        "cites": "MATCH (:Paper)-[r:CITES]->(:Paper) RETURN count(r) AS n",
        "cites_added": "MATCH ()-[r:CITES_ADDED]->() RETURN count(r) AS n",
        "coupled_with": "MATCH ()-[r:COUPLED_WITH]->() RETURN count(r) AS n",
        "external_papers": "MATCH (e:ExternalPaper) RETURN count(e) AS n",
        "cites_external": "MATCH ()-[r:CITES_EXTERNAL]->() RETURN count(r) AS n",
    }
    with driver.session() as session:
        return {name: session.run(query).single()["n"] for name, query in queries.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("fetch-s2", "build", "fetch-external", "write", "stats"))
    parser.add_argument("--write", action="store_true", help="required to modify Neo4j")
    parser.add_argument("--min-citing", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    corpus = load_corpus()
    ids = {p["id"] for p in corpus}

    if args.stage == "fetch-s2":
        missing = [p["id"] for p in corpus if not parse_refs(p.get("references"))]
        print(f"{len(missing)} corpus papers have no stored reference list", flush=True)
        _fetch(missing, "paperId,references.paperId", SUPPLEMENT,
               keep=lambda p: [r["paperId"] for r in p.get("references") or [] if r and r.get("paperId")])
        return 0

    supplement = json.loads(SUPPLEMENT.read_text()) if SUPPLEMENT.exists() else {}
    supplement.pop("_attempted", None)
    refs, added = reference_sets(corpus, supplement)

    if args.stage == "build":
        coupled = coupling_edges(refs, top_k=args.top_k)
        closure = closure_targets(refs, ids, min_citing=args.min_citing)
        plan = {
            "settings": {"coupling_top_k": args.top_k, "coupling_min_shared": 2, "coupling_max_ref_df": 1000,
                         "closure_min_citing": args.min_citing},
            "papers_with_references": sum(bool(v) for v in refs.values()),
            "supplement_papers": len(added),
            "cites_added": sorted((s, t) for s, values in added.items() for t in values if t in ids and t != s),
            "coupled_with": coupled,
            "closure": closure,
            "cites_external": sorted((s, t) for s, values in refs.items() for t in values if t in closure),
        }
        PLAN.write_text(json.dumps(plan))
        print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in plan.items() if k != "settings"}))
        return 0

    plan = json.loads(PLAN.read_text())
    if args.stage == "fetch-external":
        _fetch(sorted(plan["closure"]), "paperId,title,abstract,year,venue,citationCount", EXTERNAL,
               keep=lambda p: {k: p.get(k) for k in ("title", "abstract", "year", "venue", "citationCount")})
        return 0

    from src.storage.neo4j_store import get_driver

    driver = get_driver()
    try:
        if args.stage == "write":
            if not args.write:
                parser.error("write requires --write")
            external = json.loads(EXTERNAL.read_text()) if EXTERNAL.exists() else {}
            external.pop("_attempted", None)
            print(json.dumps(write_graph(driver, plan, external)))
        print(json.dumps(graph_stats(driver), indent=2))
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
