"""Densify the citation graph without changing the base corpus or its CITES edges.

Only 10,899 of 47,619 papers came with reference lists (Semantic Scholar hides
most lists for publisher-licensing reasons), and only 3.6% of references resolve
inside the corpus, so 71% of papers have no citation edge. This module adds
three opt-in layers. Each uses its own label or relationship type, so the
original graph, retrieval defaults, and frozen evaluations are unchanged until a
variant enables them (``GRAPH_EXPANSION_EDGES``):

* ``CITES_ADDED``: corpus-internal citations found in reference lists fetched later;
* ``COUPLED_WITH``: bibliographic coupling between corpus papers that share
  references (Kessler, 1963), cosine-normalised, top-k per paper;
* ``(:ExternalPaper)`` with ``CITES_EXTERNAL``: the papers outside the corpus
  most cited by it (citation closure), for context and hypothesis evidence.
  They never carry the ``Paper`` label, so retrieval cannot return them.

Reference data: the Semantic Scholar supplement (``data/s2_references_supplement.json``)
plus OpenAlex (CC0), matched by DOI (``OPENALEX_API_KEY`` in ``.env``).

Stages: ``fetch-s2`` and ``fetch-openalex`` -> ``build`` -> ``fetch-external`` ->
``write --write`` -> ``stats``.
Undo: delete ``CITES_ADDED``/``COUPLED_WITH``/``CITES_EXTERNAL`` relationships and
``ExternalPaper`` nodes; base data is never touched.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

SUPPLEMENT = Path("data/s2_references_supplement.json")
OPENALEX = Path("data/openalex_references.json")
EXTERNAL = Path("data/openalex_external_papers.json")
PLAN = Path("data/citation_densify_plan.json")
OPENALEX_URL = "https://api.openalex.org/works"


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
    import numpy as np
    from scipy.sparse import csr_matrix

    papers = sorted(p for p, values in refs.items() if values)
    df = Counter(ref for p in papers for ref in refs[p])
    columns = {ref: i for i, ref in enumerate(sorted(r for r, n in df.items() if 2 <= n <= max_ref_df))}
    rows, cols = [], []
    for row, paper in enumerate(papers):
        for ref in refs[paper]:
            if ref in columns:
                rows.append(row)
                cols.append(columns[ref])
    matrix = csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(len(papers), len(columns)))
    shared = (matrix @ matrix.T).tocsr()  # sparse: shared-reference counts
    lengths = np.array([len(refs[p]) for p in papers], dtype=float)
    best = {}
    for a in range(len(papers)):
        start, end = shared.indptr[a], shared.indptr[a + 1]
        neighbours, counts = shared.indices[start:end], shared.data[start:end]
        keep = (neighbours != a) & (counts >= min_shared)
        neighbours, counts = neighbours[keep], counts[keep]
        if not len(neighbours):
            continue
        scores = counts / np.sqrt(lengths[a] * lengths[neighbours])
        for index in np.argsort(-scores, kind="stable")[:top_k]:
            b, n, score = int(neighbours[index]), int(counts[index]), float(scores[index])
            key = (papers[a], papers[b]) if papers[a] < papers[b] else (papers[b], papers[a])
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


def _openalex_get(params):
    import requests
    from dotenv import load_dotenv

    load_dotenv(".env")
    key = os.getenv("OPENALEX_API_KEY", "").strip().strip('"')
    query = dict(params, api_key=key) if key else dict(params)
    for attempt in range(8):
        try:
            response = requests.get(OPENALEX_URL, params=query, headers={"User-Agent": "research-client/1.0"},
                                    timeout=60)
        except requests.RequestException:
            time.sleep(min(60.0, 2.0 ** (attempt + 1)))
            continue
        if response.status_code == 200:
            time.sleep(0.12)
            return response.json()["results"]
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(60.0, 2.0 ** (attempt + 1)))
            continue
        raise RuntimeError(f"OpenAlex error {response.status_code}: {response.text[:300]}")
    raise RuntimeError("OpenAlex request failed after retries")


def _checkpointed(path: Path, items, size, fetch_batch):
    done = json.loads(path.read_text()) if path.exists() else {}
    attempted = set(done.pop("_attempted", []))
    todo = [i for i in items if i not in attempted]
    for start in range(0, len(todo), size):
        batch = todo[start:start + size]
        done.update(fetch_batch(batch))
        attempted.update(batch)
        path.write_text(json.dumps({**done, "_attempted": sorted(attempted)}))
        if (start // size) % 20 == 0 or start + size >= len(todo):
            print(f"fetched {min(start + size, len(todo))}/{len(todo)} (kept {len(done)})", flush=True)
    return done


def fetch_openalex_references(corpus):
    """Map corpus papers to OpenAlex by DOI; keep the OpenAlex id and referenced works."""
    by_doi = {}
    for paper in corpus:
        doi = (paper.get("doi") or "").strip().lower()
        if doi and "|" not in doi and "," not in doi:
            by_doi.setdefault(doi, paper["id"])

    def batch_fetch(dois):
        found = {}
        for work in _openalex_get({"filter": "doi:" + "|".join(dois), "per-page": 50,
                                   "select": "id,doi,referenced_works"}):
            doi = (work.get("doi") or "").lower().replace("https://doi.org/", "")
            if doi in by_doi:
                found[by_doi[doi]] = {"openalex_id": work["id"].rsplit("/", 1)[-1],
                                      "refs": [r.rsplit("/", 1)[-1] for r in work.get("referenced_works") or []]}
        return found

    return _checkpointed(OPENALEX, sorted(by_doi), 50, batch_fetch)


def abstract_from_index(index):
    if not index:
        return ""
    words = sorted((position, word) for word, positions in index.items() for position in positions)
    return " ".join(word for _, word in words)


def fetch_openalex_works(work_ids):
    def batch_fetch(ids):
        works = _openalex_get({"filter": "openalex:" + "|".join(ids), "per-page": 50,
                               "select": "id,display_name,publication_year,cited_by_count,abstract_inverted_index"})
        return {work["id"].rsplit("/", 1)[-1]: {
                    "title": work.get("display_name") or "", "year": work.get("publication_year"),
                    "citationCount": work.get("cited_by_count"),
                    "abstract": abstract_from_index(work.get("abstract_inverted_index"))} for work in works}

    return _checkpointed(EXTERNAL, list(work_ids), 50, batch_fetch)


def build_plan(corpus, supplement, openalex, *, top_k=10, min_citing=10):
    """Combine Semantic Scholar and OpenAlex reference data into the three opt-in layers.

    Coupling is computed separately within each id namespace (S2 ids, OpenAlex ids)
    and merged by keeping the higher score; closure uses OpenAlex ids only, so one
    external paper never appears under two ids.
    """
    ids = {p["id"] for p in corpus}
    refs_s2, added_s2 = reference_sets(corpus, supplement)
    original = {(p["id"], t) for p in corpus for t in parse_refs(p.get("references")) if t in ids}
    oa_to_s2 = {v["openalex_id"]: k for k, v in openalex.items()}
    refs_oa = {k: set(v["refs"]) for k, v in openalex.items() if v["refs"]}
    internal = {(s, t) for s, values in added_s2.items() for t in values if t in ids}
    internal |= {(s, oa_to_s2[r]) for s, values in refs_oa.items() for r in values if r in oa_to_s2}
    cites_added = sorted((s, t) for s, t in internal if s != t and (s, t) not in original)
    coupled = {}
    for namespace, refs in (("s2", refs_s2), ("openalex", refs_oa)):
        for edge in coupling_edges({k: v for k, v in refs.items() if v}, top_k=top_k):
            key = (edge["source"], edge["target"])
            if key not in coupled or coupled[key]["score"] < edge["score"]:
                coupled[key] = {**edge, "namespace": namespace}
    closure = closure_targets(refs_oa, set(oa_to_s2), min_citing=min_citing)
    covered = set(refs_oa) | {k for k, v in refs_s2.items() if v}
    return {
        "settings": {"coupling_top_k": top_k, "coupling_min_shared": 2, "coupling_max_ref_df": 1000,
                     "closure_min_citing": min_citing, "closure_namespace": "openalex"},
        "papers_with_references": len(covered),
        "papers_with_references_before": sum(bool(parse_refs(p.get("references"))) for p in corpus),
        "openalex_matched": len(openalex),
        "cites_added": cites_added,
        "coupled_with": sorted(coupled.values(), key=lambda e: (e["source"], e["target"])),
        "closure": closure,
        "cites_external": sorted((s, r) for s, values in refs_oa.items() for r in values if r in closure),
    }


def write_graph(driver, plan, external, *, batch=5000):
    with driver.session() as session:
        session.run("CREATE INDEX paper_id_lookup IF NOT EXISTS FOR (p:Paper) ON (p.id)").consume()
        session.run("CREATE CONSTRAINT external_paper_id IF NOT EXISTS FOR (e:ExternalPaper) REQUIRE e.id IS UNIQUE").consume()
        added = [{"source": s, "target": t} for s, t in plan["cites_added"]]
        for start in range(0, len(added), batch):
            session.run("UNWIND $rows AS row MATCH (a:Paper {id: row.source}) MATCH (b:Paper {id: row.target}) "
                        "MERGE (a)-[r:CITES_ADDED]->(b) SET r.source = 'reference_supplement'",
                        rows=added[start:start + batch]).consume()
        coupled = plan["coupled_with"]
        for start in range(0, len(coupled), batch):
            session.run("UNWIND $rows AS row MATCH (a:Paper {id: row.source}) MATCH (b:Paper {id: row.target}) "
                        "MERGE (a)-[r:COUPLED_WITH]->(b) SET r.shared = row.shared, r.score = row.score, "
                        "r.namespace = row.namespace, r.source = 'bibliographic_coupling'",
                        rows=coupled[start:start + batch]).consume()
        nodes = [{"id": pid, **meta, "corpus_citations": plan["closure"][pid]}
                 for pid, meta in external.items() if pid in plan["closure"]]
        for start in range(0, len(nodes), batch):
            session.run("UNWIND $rows AS row MERGE (e:ExternalPaper {id: row.id}) "
                        "SET e.title = row.title, e.abstract = row.abstract, e.year = row.year, "
                        "e.source = 'openalex', e.citation_count = row.citationCount, "
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


def _load(path: Path):
    if not path.exists():
        return {}
    return {k: v for k, v in json.loads(path.read_text()).items() if k != "_attempted"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("fetch-s2", "fetch-openalex", "build", "fetch-external", "write", "stats"))
    parser.add_argument("--write", action="store_true", help="required to modify Neo4j")
    parser.add_argument("--min-citing", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    corpus = load_corpus()

    if args.stage == "fetch-s2":
        missing = [p["id"] for p in corpus if not parse_refs(p.get("references"))]
        print(f"{len(missing)} corpus papers have no stored reference list", flush=True)
        _fetch(missing, "paperId,references.paperId", SUPPLEMENT,
               keep=lambda p: [r["paperId"] for r in p.get("references") or [] if r and r.get("paperId")])
        return 0
    if args.stage == "fetch-openalex":
        fetch_openalex_references(corpus)
        return 0
    if args.stage == "build":
        plan = build_plan(corpus, _load(SUPPLEMENT), _load(OPENALEX), top_k=args.top_k, min_citing=args.min_citing)
        PLAN.write_text(json.dumps(plan))
        print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in plan.items() if k != "settings"}))
        return 0

    plan = json.loads(PLAN.read_text())
    if args.stage == "fetch-external":
        fetch_openalex_works(sorted(plan["closure"]))
        return 0

    from src.storage.neo4j_store import get_driver

    driver = get_driver()
    try:
        if args.stage == "write":
            if not args.write:
                parser.error("write requires --write")
            print(json.dumps(write_graph(driver, plan, _load(EXTERNAL))))
        print(json.dumps(graph_stats(driver), indent=2))
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
