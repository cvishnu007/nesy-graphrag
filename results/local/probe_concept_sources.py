"""Probe concept-graph variants (set via env before import) on the 26 probe queries.

Usage: CONCEPT_SOURCE=... CONCEPT_EXCLUDE_GENERIC=... python probe_concept_sources.py <label>
"""
import contextlib
import io
import json
import statistics
import sys

sys.path.insert(0, ".")
from src.pipeline.contradiction import detect_contradictions  # noqa: E402
from src.pipeline.hypothesis import generate_hypotheses  # noqa: E402
from src.pipeline.metrics import compute_hns  # noqa: E402
from src.storage.neo4j_store import get_driver  # noqa: E402
from src.utils import config  # noqa: E402

label = sys.argv[1]
bench = json.load(open("evaluation/benchmarks/retrieval_queries.json"))
queries = [q["query"] for q in bench["queries"]] + [
    "cybersecurity", "databases", "software engineering", "cloud computing", "operating systems", "computer networks"]
driver = get_driver()
rows = []
for query in queries:
    with contextlib.redirect_stdout(io.StringIO()):
        pairs = detect_contradictions(driver, query, top_k=25)
        hypotheses = generate_hypotheses(driver, query, top_k=25)
    rows.append({"query": query, "contradiction_pairs": len(pairs), "hypothesis_candidates": len(hypotheses),
                 "hypothesis_top_shared": hypotheses[0]["shared_concept_names"][:5] if hypotheses else [],
                 "hypothesis_top_score": hypotheses[0]["evidence_score"] if hypotheses else None,
                 "contradiction_top_shared": pairs[0]["shared_concept_names"][:5] if pairs else []})
    print(json.dumps(rows[-1]), flush=True)
node, rel = ("SciEntity", "MENTIONS") if config.CONCEPT_SOURCE == "scientific" else ("Concept", "RELATED_TO")
ids = [r["id"] for r in driver.execute_query(
    f"MATCH (p:Paper)-[:{rel}]->(c:{node}) WHERE NOT coalesce(c.generic,false) "
    "WITH DISTINCT p, rand() AS x ORDER BY x LIMIT 24 RETURN p.id AS id").records]
hns = []
for i in range(0, len(ids) - 1, 2):
    with contextlib.redirect_stdout(io.StringIO()):
        result = compute_hns(driver, [{"paper": {"id": ids[i]}}], [ids[i + 1]])["individual_scores"]
    hns.append(result[0] if result else None)
driver.close()
summary = {
    "label": label, "concept_source": config.CONCEPT_SOURCE, "exclude_generic": config.CONCEPT_EXCLUDE_GENERIC,
    "queries": len(rows),
    "contradiction_pairs_mean": statistics.mean(r["contradiction_pairs"] for r in rows),
    "contradiction_zero_queries": sum(r["contradiction_pairs"] == 0 for r in rows),
    "hypothesis_candidates_mean": statistics.mean(r["hypothesis_candidates"] for r in rows),
    "hypothesis_zero_queries": sum(r["hypothesis_candidates"] == 0 for r in rows),
    "hns_values": hns, "hns_distinct": sorted({round(v, 3) for v in hns if v is not None}),
}
print("SUMMARY", json.dumps(summary), flush=True)
json.dump({"summary": summary, "rows": rows},
          open(f"results/local/concept_probe_{label}_20261114.json", "w", encoding="utf-8"), indent=1)
