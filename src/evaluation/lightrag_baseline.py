"""LightRAG-style retrieval baseline over the SciERC entity graph (full corpus).

A real LightRAG index needs about two LLM extraction calls per chunk, which for
47,619 abstracts on a 4 GB laptop GPU is well over a week. This baseline keeps
LightRAG's query-time algorithm (Guo et al., 2024; query logic of LightRAG v1.4)
and replaces only the LLM-built graph with the graph we already have:

* Entities: every SciERC entity mention in the corpus (supervised tagger, F1 0.64),
  embedded by name with BGE. LightRAG embeds an LLM-written description; we have
  none, so the vector text is the entity name.
* Relations: entity pairs that co-occur in at least ``MIN_COOCCURRENCE`` abstracts,
  embedded as "source, target". LightRAG relations come with LLM keywords and a
  description; ours are untyped co-occurrence.
* Chunks: one abstract per paper, so a retrieved chunk is a retrieved paper.

Query time, as in LightRAG:

1. The LLM (local qwen2.5-7B) extracts high-level and low-level keywords with
   LightRAG's keyword-extraction instructions; results are cached.
2. local: top ``TOP_K`` entities by cosine to the joined low-level keywords; each
   entity contributes its papers, ordered by (entity rank, -number of the entity's
   one-hop neighbours mentioned in the paper), as in
   ``_find_most_related_text_unit_from_entities``.
3. global: top ``TOP_K`` relations by cosine to the joined high-level keywords,
   re-sorted by (edge degree, weight) as in ``_get_edge_data``; each relation
   contributes the papers mentioning both entities.
4. hybrid = round-robin merge of local and global; mix = round-robin merge of
   vector (BGE), local and global (LightRAG's default ``mix`` mode).

Stages (resumable): ``build`` (index vectors, about 10 minutes on GPU),
``keywords`` (44 LLM calls), ``rank``, ``grade`` (pool extension with the same
validated 7B judge) and ``score``. Results go to a new directory; the frozen
stratified benchmark directory is only read.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

NER_FILE = Path("data/s2_ner_scientific.json")
INDEX_DIR = Path("data/lightrag_style")
FROZEN = Path("results/retrieval/stratified_v2_20261114")
OUTPUT = Path("results/retrieval/stratified_v2_lightrag_20261119")
QUERIES = "evaluation/benchmarks/retrieval_queries_v2_stratified_draft.json"
TOP_K = 60
MIN_COOCCURRENCE = 2
DEPTH = 20
VARIANTS = ("lightrag_local", "lightrag_global", "lightrag_hybrid", "lightrag_mix")
GRADED_VARIANTS = ("lightrag_hybrid", "lightrag_mix")
COMPARISONS = [
    ("full_bge_bm25_similar", "lightrag_hybrid"),
    ("full_bge_bm25_similar", "lightrag_mix"),
    ("rrf_bge_bm25", "lightrag_hybrid"),
    ("rrf_bge_bm25", "lightrag_mix"),
    ("bge", "lightrag_mix"),
    ("lightrag_mix", "lightrag_hybrid"),
]

# Adapted from LightRAG's PROMPTS["keywords_extraction"] (role, goal, instructions, JSON output).
KEYWORD_PROMPT = """---Role---
You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---
Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---
- Output the keywords in JSON format, it will be parsed by a JSON parser, do not add any extra content in output.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes
  - "low_level_keywords" for specific entities or details

---Example---
Query: "How does international trade influence global economic stability?"
Output: {"high_level_keywords": ["International trade", "Global economic stability", "Economic impact"], "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]}

---Real Data---
Query: "{query}"
Output:"""


def normalise(name):
    return re.sub(r"\s+", " ", name.strip().lower())


def load_entities():
    records = json.loads(NER_FILE.read_text(encoding="utf-8"))["records"]
    papers, mentions, types = [], [], defaultdict(Counter)
    for record in records:
        names = set()
        for entity in record["entities"]:
            name = normalise(entity["text"])
            if name:
                names.add(name)
                types[name][entity["type"]] += 1
        papers.append(record["id"])
        mentions.append(names)
    return papers, mentions, {n: c.most_common(1)[0][0] for n, c in types.items()}


def encode(texts, path, batch_size=256, chunk=50_000):
    """Write float16 BGE vectors to ``path`` in chunks, so RAM holds one chunk at a time."""
    from src.storage.chroma_store import get_embedder

    model = get_embedder()
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(texts), 768))
    for start in range(0, len(texts), chunk):
        out[start:start + chunk] = model.encode(texts[start:start + chunk], batch_size=batch_size,
                                                normalize_embeddings=True, convert_to_numpy=True).astype(np.float16)
        print(f"  {path.name}: {min(start + chunk, len(texts))}/{len(texts)}", flush=True)
    out.flush()
    del out


def build():
    """Entity and relation tables plus their BGE vectors (float16, stored as .npy)."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    papers, mentions, etype = load_entities()
    entities = sorted(etype)
    index = {e: i for i, e in enumerate(entities)}
    paper_entities = [sorted(index[n] for n in names) for names in mentions]
    co = Counter()
    for ids in paper_entities:
        co.update(combinations(ids, 2))
    relations = sorted((a, b, w) for (a, b), w in co.items() if w >= MIN_COOCCURRENCE)
    print(f"{len(papers)} papers, {len(entities)} entities, {len(relations)} relations", flush=True)
    (INDEX_DIR / "graph.json").write_text(json.dumps({
        "papers": papers, "entities": entities, "types": [etype[e] for e in entities],
        "paper_entities": paper_entities, "relations": relations,
        "min_cooccurrence": MIN_COOCCURRENCE, "ner_file": str(NER_FILE)}))
    encode(entities, INDEX_DIR / "entity_vectors.npy")
    encode([f"{entities[a]}, {entities[b]}" for a, b, _ in relations], INDEX_DIR / "relation_vectors.npy")


class LightRAGStyle:
    def __init__(self):
        graph = json.loads((INDEX_DIR / "graph.json").read_text())
        self.papers = graph["papers"]
        self.entities = graph["entities"]
        self.paper_entities = [set(ids) for ids in graph["paper_entities"]]
        self.relations = graph["relations"]
        self.entity_vectors = np.load(INDEX_DIR / "entity_vectors.npy", mmap_mode="r")
        self.relation_vectors = np.load(INDEX_DIR / "relation_vectors.npy", mmap_mode="r")
        self.entity_papers = defaultdict(list)
        for p, ids in enumerate(self.paper_entities):
            for e in ids:
                self.entity_papers[e].append(p)
        self.neighbours = defaultdict(set)
        self.relation_papers_index = {}
        for a, b, _ in self.relations:
            self.neighbours[a].add(b)
            self.neighbours[b].add(a)
        self.degree = {e: len(n) for e, n in self.neighbours.items()}

    @staticmethod
    def _top(vectors, query_vector, k, chunk=200_000):
        scores = np.concatenate([np.asarray(vectors[i:i + chunk], dtype=np.float32) @ query_vector
                                 for i in range(0, len(vectors), chunk)])
        top = np.argpartition(-scores, k)[:k]
        return top[np.argsort(-scores[top])]

    def _vector(self, text):
        from src.storage.chroma_store import get_embedder

        return get_embedder().encode([text], normalize_embeddings=True)[0].astype(np.float32)

    def local(self, keywords):
        ranked_entities = self._top(self.entity_vectors, self._vector(keywords), TOP_K)
        order = []
        for e in ranked_entities:
            e = int(e)
            near = self.neighbours.get(e, set())
            papers = self.entity_papers[e]
            order += sorted(papers, key=lambda p: -len(self.paper_entities[p] & near))
        return self._unique(order)

    def global_(self, keywords):
        ranked = [self.relations[int(i)] for i in self._top(self.relation_vectors, self._vector(keywords), TOP_K)]
        ranked.sort(key=lambda r: (-(self.degree.get(r[0], 0) + self.degree.get(r[1], 0)), -r[2]))
        order = []
        for a, b, _ in ranked:
            order += sorted(set(self.entity_papers[a]) & set(self.entity_papers[b]))
        return self._unique(order)

    def _unique(self, order):
        seen, out = set(), []
        for p in order:
            if p not in seen:
                seen.add(p)
                out.append(self.papers[p])
        return out

    @staticmethod
    def round_robin(*lists, depth=DEPTH):
        seen, out = set(), []
        for position in range(max(map(len, lists), default=0)):
            for ranked in lists:
                if position < len(ranked) and ranked[position] not in seen:
                    seen.add(ranked[position])
                    out.append(ranked[position])
        return out[:depth]

    def rank(self, query, keywords, vector_ids):
        low = ", ".join(keywords.get("low_level_keywords") or keywords.get("high_level_keywords") or [query])
        high = ", ".join(keywords.get("high_level_keywords") or keywords.get("low_level_keywords") or [query])
        local, global_ = self.local(low)[:DEPTH * 3], self.global_(high)[:DEPTH * 3]
        return {"lightrag_local": local[:DEPTH], "lightrag_global": global_[:DEPTH],
                "lightrag_hybrid": self.round_robin(local, global_),
                "lightrag_mix": self.round_robin(vector_ids, local, global_)}


def parse_keywords(text):
    match = re.search(r"\{.*\}", text, re.S)
    data = json.loads(match.group(0)) if match else {}
    return {key: [str(k).strip() for k in data.get(key, []) if str(k).strip()]
            for key in ("high_level_keywords", "low_level_keywords")}


def extract_keywords(queries, path):
    from src.pipeline.orchestrator import get_llm

    done = json.loads(path.read_text()) if path.exists() else {}
    client = get_llm()
    for query in queries:
        if query["query_id"] in done:
            continue
        raw = client.complete(KEYWORD_PROMPT.replace("{query}", query["query"]), max_tokens=300, temperature=0.0)
        done[query["query_id"]] = {"query": query["query"], "raw": raw, **parse_keywords(raw)}
        path.write_text(json.dumps(done, indent=2))
        print(query["query_id"], done[query["query_id"]]["low_level_keywords"], flush=True)
    return done


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("build", "keywords", "rank", "grade", "score"))
    parser.add_argument("--judge", default="ollama:qwen2.5:7b-instruct@strict")
    args = parser.parse_args(argv)
    if args.stage == "build":
        build()
        return 0
    from src.evaluation import stratified_benchmark as sb

    queries = json.loads(Path(QUERIES).read_text(encoding="utf-8"))["queries"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    keyword_path = OUTPUT / "keywords.json"
    if args.stage == "keywords":
        extract_keywords(queries, keyword_path)
        return 0
    rankings_path = OUTPUT / "rankings.jsonl"
    if args.stage == "rank":
        keywords = json.loads(keyword_path.read_text())
        frozen = sb.load_rankings(FROZEN)
        model = LightRAGStyle()
        with rankings_path.open("w") as stream:
            for row in map(json.loads, (FROZEN / "rankings.jsonl").read_text().splitlines()):
                stream.write(json.dumps(row) + "\n")
            for query in queries:
                ranked = model.rank(query["query"], keywords[query["query_id"]], frozen["bge"][query["query_id"]])
                for name in VARIANTS:
                    stream.write(json.dumps({"variant": name, "family": "lightrag_style", "query_id": query["query_id"],
                                             "ranked_paper_ids": ranked[name]}) + "\n")
                print(query["query_id"], {n: len(ranked[n]) for n in VARIANTS}, flush=True)
        return 0

    rankings = sb.load_rankings(OUTPUT)
    from src.utils.config import CLEAN_FILE

    corpus = {p["id"]: p for p in json.loads(Path(CLEAN_FILE).read_text(encoding="utf-8"))}
    graded_variants = {n: r for n, r in rankings.items() if n not in VARIANTS or n in GRADED_VARIANTS}
    rows = sb.pool(queries, graded_variants, corpus)
    labels_path = OUTPUT / "labels.jsonl"
    if not labels_path.exists():
        shutil.copy(FROZEN / "labels.jsonl", labels_path)
    if args.stage == "grade":
        from src.evaluation.judge_agreement import judge, make_client

        model_spec, prompt_name = args.judge.rsplit("@", 1)
        known = {r["item_id"] for r in map(json.loads, labels_path.read_text().splitlines())}
        print(f"{len(rows)} pooled pairs, {sum(r['item_id'] not in known for r in rows)} new to grade", flush=True)
        judge(make_client(model_spec), rows, prompt_name, labels_path, batch_size=1)
        (OUTPUT / "judge.json").write_text(json.dumps({
            "judge": args.judge, "batch_size": 1, "pool_depth": DEPTH, "pooled_pairs": len(rows),
            "frozen_labels_reused_from": str(FROZEN / "labels.jsonl"),
            "human_validation": "results/retrieval/judge_agreement_local_20261114/report.json (kappa 0.7485)"}, indent=2) + "\n")
        return 0

    from src.evaluation.regrade_pool import score
    from src.evaluation.significance import paired_permutation_test, paired_significance

    graded = {r["item_id"]: r["grade"] for r in map(json.loads, labels_path.read_text().splitlines())}
    labels = defaultdict(dict)
    for row in rows:
        labels[row["query_id"]][row["paper_id"]] = graded[row["item_id"]]
    report = {"note": "LightRAG-style baseline; graded variants are fully judged at depth 20. "
                      "lightrag_local/global are reported with unjudged papers counted as non-relevant.",
              "graded_variants": list(GRADED_VARIANTS), "top_k": TOP_K, "min_cooccurrence": MIN_COOCCURRENCE,
              "splits": {}}
    for split in ("all", "dev", "test"):
        ids = sorted(q["query_id"] for q in queries if split == "all" or q["split"] == split)
        summary = score(rankings, labels, ids)
        comparisons = {}
        for challenger, reference in COMPARISONS:
            a, b = summary[challenger]["per_query_ndcg@10"], summary[reference]["per_query_ndcg@10"]
            deltas = [a[q] - b[q] for q in ids]
            test = paired_significance(deltas) if len(deltas) <= 20 else paired_permutation_test(deltas)
            comparisons[f"{challenger}_vs_{reference}"] = {**test, "wins": sum(d > 0 for d in deltas),
                                                           "losses": sum(d < 0 for d in deltas)}
        report["splits"][split] = {"query_count": len(ids), "variants": summary, "comparisons": comparisons}
        print(f"\n{split} ({len(ids)} queries)")
        for name, data in sorted(summary.items(), key=lambda item: -item[1]["ndcg@10"]):
            print(f"  {name:28} ndcg@10={data['ndcg@10']:.3f} p@10={data['precision@10']:.3f} "
                  f"unjudged@20={data['unjudged_rate@20']:.3f}")
        for name, data in comparisons.items():
            p = data.get("exact_two_sided_randomization_p", data.get("monte_carlo_two_sided_p"))
            print(f"  {name:50} delta={data['mean_delta']:+.3f} p={p:.4f} W/L={data['wins']}/{data['losses']}")
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
