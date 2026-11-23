"""Calibrate the query gate (src/pipeline/query_gate.py) on valid and invalid queries.

Valid: the 44 topic-stratified benchmark queries, the 20 GNN benchmark queries, the
demo queries in docs/DEMO_QUERIES.md, and short topic names a user might type.
Invalid: everyday and conversational sentences with no computer-science research
content, written for this check (fixed list below).
Out-of-domain science (biology, physics, medicine...) is reported separately: those
are scientific, but this corpus has no papers for them.

For each query the gate records its closeness (mean BGE cosine similarity of the 5
nearest papers) and the multi-word SciERC entities it names, and applies the full
gate rule (src/pipeline/query_gate.py). Needs Chroma and Neo4j.

    python -m src.evaluation.query_gate_calibration
"""
from __future__ import annotations

import json
import re
from pathlib import Path

OUT = Path("results/query_gate_20261123")

SHORT_TOPICS = [
    "machine learning", "cyber security", "computer vision", "databases", "cloud computing",
    "operating systems", "natural language processing", "software engineering", "data science",
    "computer networks", "graph neural networks", "deep learning", "artificial intelligence",
    "large language models", "intrusion detection", "federated learning",
]

INVALID = [
    "hello how are you", "hi", "good morning", "what is your name", "tell me a joke",
    "what is the weather today", "will it rain tomorrow in bangalore", "best pizza near me",
    "how to cook biryani at home", "recipe for chocolate cake", "who won the cricket match yesterday",
    "ipl schedule this year", "latest bollywood movies", "suggest a good song to listen to",
    "i am feeling bored", "i am hungry", "what should i wear to a wedding", "how to lose weight fast",
    "best places to visit in goa", "cheap flights to delhi", "how much does a car cost",
    "what time is it", "how old is the earth", "tell me about your family", "my cat is sleeping",
    "the sky is blue and the grass is green", "i like ice cream", "let us go to the beach",
    "how to tie a tie", "what is the capital of france", "how to make friends in college",
    "is it going to be sunny this weekend", "how to get a girlfriend", "funny memes",
    "write a poem about love", "happy birthday to you", "why is my phone battery draining",
    "how do i open a bank account", "train timings from bangalore to mysore", "asdfgh qwerty",
]

OUT_OF_DOMAIN_SCIENCE = [
    "photosynthesis in plants", "black hole formation", "protein folding in cells",
    "climate change and sea level rise", "treatment of type 2 diabetes", "theory of evolution",
    "chemical bonding in organic molecules", "plate tectonics and earthquakes",
]


def demo_queries():
    text = Path("docs/DEMO_QUERIES.md").read_text(encoding="utf-8")
    return sorted({re.sub(r"\s*\(.*\)$", "", line[2:].strip()) for line in text.splitlines() if line.startswith("- ")})


def benchmark_queries():
    strat = json.loads(Path("evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json").read_text(encoding="utf-8"))
    gnn = json.loads(Path("results/retrieval/expanded_20260913/benchmark.json").read_text(encoding="utf-8"))
    return [q["query"] for q in strat["queries"]] + [q["query"] for q in gnn["queries"]]


def main() -> int:
    from src.pipeline.query_gate import content_words, gate_decision, scientific_terms
    from src.storage.chroma_store import query as chroma_query
    from src.storage.neo4j_store import get_driver
    from src.utils.config import QUERY_GATE_ENTITY_MIN_SIMILARITY, QUERY_GATE_MIN_SIMILARITY, QUERY_GATE_TOP_N

    driver = get_driver()

    groups = {"valid_benchmark": benchmark_queries(), "valid_demo": demo_queries(), "valid_short_topic": SHORT_TOPICS,
              "invalid_everyday": INVALID, "out_of_domain_science": OUT_OF_DOMAIN_SCIENCE}
    rows = []
    for group, queries in groups.items():
        for q in queries:
            sims = [p["score"] for p in chroma_query(q, top_k=QUERY_GATE_TOP_N)]
            terms = scientific_terms(driver, q)
            decision = gate_decision(q, sims, terms)
            rows.append({"group": group, "query": q, "top1": round(sims[0], 4),
                         "mean_top5": round(sum(sims) / len(sims), 4), "has_content_words": bool(content_words(q)),
                         "scientific_terms": terms, "valid": decision["valid"], "reason": decision["reason"]})
    driver.close()

    def stats(group):
        vals = sorted(r["mean_top5"] for r in rows if r["group"] == group)
        return {"n": len(vals), "min": vals[0], "max": vals[-1], "median": vals[len(vals) // 2]}

    valid = [r for r in rows if r["group"].startswith("valid")]
    invalid = [r for r in rows if r["group"] == "invalid_everyday"]
    def passes(r):
        return r["valid"]

    summary = {
        "statistic": f"mean cosine similarity of the {QUERY_GATE_TOP_N} nearest papers (BGE, query prefix)",
        "rule": (f"no content words -> invalid; closeness >= {QUERY_GATE_MIN_SIMILARITY} -> valid; closeness >= "
                 f"{QUERY_GATE_ENTITY_MIN_SIMILARITY} with a multi-word SciERC entity (df >= 5) -> valid; else invalid"),
        "groups": {g: stats(g) for g in groups},
        "valid_passed": f"{sum(passes(r) for r in valid)}/{len(valid)}",
        "invalid_rejected": f"{sum(not passes(r) for r in invalid)}/{len(invalid)}",
        "out_of_domain_rejected": f"{sum(not passes(r) for r in rows if r['group'] == 'out_of_domain_science')}/{len(OUT_OF_DOMAIN_SCIENCE)}",
        "lowest_valid": sorted(valid, key=lambda r: r["mean_top5"])[:5],
        "highest_invalid": sorted(invalid, key=lambda r: -r["mean_top5"])[:5],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("rule", "groups", "valid_passed", "invalid_rejected",
                                                "out_of_domain_rejected")}, indent=2))
    for label in ("lowest_valid", "highest_invalid"):
        print(label)
        for r in summary[label]:
            print(f"  {r['mean_top5']:.4f}  {r['reason']:18} {r['query']}  {r['scientific_terms']}")
    print("out-of-domain science:")
    for r in rows:
        if r["group"] == "out_of_domain_science":
            print(f"  {r['mean_top5']:.4f}  {r['reason']:18} {r['query']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
