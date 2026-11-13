"""Probe: NLI sentence-level contradiction candidates on the 26 probe queries (no LLM)."""
import contextlib
import io
import json
import statistics
import sys
import time

sys.path.insert(0, ".")
from src.pipeline.contradiction_nli import rank_contradiction_candidates  # noqa: E402
from src.pipeline.retrieval import nesy_retrieve  # noqa: E402
from src.storage.neo4j_store import get_driver  # noqa: E402

bench = json.load(open("evaluation/benchmarks/retrieval_queries.json"))
queries = [q["query"] for q in bench["queries"]] + [
    "cybersecurity", "databases", "software engineering", "cloud computing", "operating systems", "computer networks"]
driver = get_driver()
rows = []
for query in queries:
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        papers = nesy_retrieve(driver, query, top_k=10)
    candidates = rank_contradiction_candidates(papers, top_k=5)
    rows.append({
        "query": query, "candidates": len(candidates),
        "scores": [c["nli_contradiction"] for c in candidates],
        "top": None if not candidates else {
            "paper1": candidates[0]["paper1"]["title"], "paper2": candidates[0]["paper2"]["title"],
            "evidence_1": candidates[0]["evidence_1"]["text"], "evidence_2": candidates[0]["evidence_2"]["text"],
            "nli_contradiction": candidates[0]["nli_contradiction"]},
        "seconds": round(time.perf_counter() - started, 1),
    })
    print(json.dumps({k: rows[-1][k] for k in ("query", "candidates", "scores", "seconds")}), flush=True)
driver.close()
top_scores = [r["scores"][0] for r in rows if r["scores"]]
summary = {
    "queries": len(rows), "queries_with_candidates": sum(r["candidates"] > 0 for r in rows),
    "mean_candidates": statistics.mean(r["candidates"] for r in rows),
    "top_score_quantiles": sorted(top_scores)[::max(1, len(top_scores) // 5)],
    "queries_with_top_score_ge_0.5": sum(s >= 0.5 for s in top_scores),
}
print("SUMMARY", json.dumps(summary), flush=True)
json.dump({"rows": rows, "summary": summary},
          open("results/local/nli_contradiction_probe_20261113.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
