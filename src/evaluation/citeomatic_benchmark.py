"""Global citation recommendation on the public DBLP and PubMed benchmarks of
Bhagavatula et al. (NAACL 2018, "Content-Based Citation Recommendation").

Data: ``comparison/<dataset>/corpus.json`` from AI2's public citeomatic release
(s3://ai2-s2-research-public/citeomatic-2018-02-12/), stored in
``data/external/citeomatic/``. The protocol follows the citeomatic code
(``corpus.py``, ``training.py``, ``eval_metrics.py``):

* Splits by year. DBLP: train 1966-2007, valid 2008, test 2009-2011.
  PubMed: train 1966-2008, valid 2009, test 2010-2013.
* Queries: valid or test papers. Candidates: training-split papers only.
* Gold: the query's out-citations that are in the corpus, no newer than the query,
  not the query itself, and in the training split. Queries with fewer than 10 gold
  citations are skipped.
* F1@20 = harmonic mean of the mean P@20 and mean R@20 over queries; MRR uses the
  whole ranked list (here 100). P@20 divides by 20 even if fewer papers are returned.

Methods (no training on these datasets):

* BM25 over training papers, query = the query paper's title and abstract.
* BGE (bge-base-en-v1.5): cosine between title-abstract document vectors.
* RRF(BGE, BM25).
* RRF(BGE, BM25, co-citation) - the configuration evaluated in RESULTS 2f: the
  query's 50 nearest training papers by BGE, and the training papers they cite,
  scored by summed neighbour similarity; RRF k = 60, graph weight 1.0. The query's
  own references are never used. Settings are fixed from 2f, not tuned here.

The query paper's own text and the training papers' reference lists are the only
inputs, so this is a content-based setting with a candidate-side citation graph;
citeomatic also trains on the training-split citation graph.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np

DATA = Path("data/external/citeomatic")
OUT = Path("results/retrieval/citeomatic_20261119")
SPLITS = {
    "dblp": {"train": (1966, 2007), "valid": (2008, 2008), "test": (2009, 2011)},
    "pubmed": {"train": (1966, 2008), "valid": (2009, 2009), "test": (2010, 2013)},
}
MIN_GOLD = 10
DEPTH = 100
NEIGHBOURS = 50
GRAPH_WEIGHT = 1.0
PUBLISHED = {  # Bhagavatula et al. 2018, Table 2 (F1@20, MRR)
    "dblp": {"BM25 (theirs)": (0.119, 0.425), "ClusCite": (0.237, 0.548), "NNSelect": (0.282, 0.579),
             "NNSelect + NNRank": (0.302, 0.672), "+ metadata": (0.303, 0.689)},
    "pubmed": {"BM25 (theirs)": (0.209, 0.574), "ClusCite": (0.274, 0.578), "NNSelect": (0.309, 0.699),
               "NNSelect + NNRank": (0.325, 0.754), "+ metadata": (0.329, 0.771)},
}


def _list(value):
    if isinstance(value, list):
        return value
    return ast.literal_eval(value) if value else []


def load(dataset):
    papers = {}
    with open(DATA / dataset / "corpus.json", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            papers[str(row["id"])] = {
                "id": str(row["id"]), "year": int(row.get("year") or 2017),
                "title": (row.get("title_raw") or row.get("title") or "").strip(),
                "abstract": (row.get("abstract_raw") or row.get("abstract") or "").strip(),
                "out": {str(c) for c in _list(row.get("out_citations"))} - {str(row["id"])},
            }
    return papers


def split_ids(papers, dataset, split):
    low, high = SPLITS[dataset][split]
    return sorted(pid for pid, p in papers.items() if low <= p["year"] <= high)


def gold(papers, pid, train):
    year = papers[pid]["year"]
    return {c for c in papers[pid]["out"] if c in papers and papers[c]["year"] <= year and c in train}


def metrics(ranked, gold_set, k=20):
    hits = [p in gold_set for p in ranked]
    first = next((i for i, h in enumerate(hits) if h), None)
    return {"p@20": sum(hits[:k]) / k, "r@20": sum(hits[:k]) / len(gold_set),
            "rr": 0.0 if first is None else 1.0 / (first + 1)}


def summarise(per_query):
    p = float(np.mean([m["p@20"] for m in per_query.values()]))
    r = float(np.mean([m["r@20"] for m in per_query.values()]))
    return {"queries": len(per_query), "p@20": round(p, 4), "r@20": round(r, 4),
            "f1@20": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "mrr": round(float(np.mean([m["rr"] for m in per_query.values()])), 4)}


def text(paper):
    return ". ".join(part for part in (paper["title"], paper["abstract"]) if part)


ENCODERS = ("bge", "specter2_base", "scincl")


def make_encoder(encoder):
    """Return (encode(texts) -> unit vectors, text(paper)) for an encoder name.

    ``bge`` is the app's embedding model (title. abstract). ``specter2_base`` and ``scincl``
    are citation-trained scientific-paper encoders, used as in RESULTS §1 (CLS pooling,
    title [SEP] abstract; SPECTER2 base without a task adapter).
    """
    if encoder == "bge":
        from src.storage.chroma_store import get_embedder

        model = get_embedder()
        return (lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True,
                                           convert_to_numpy=True).astype(np.float32)), text
    import torch
    from src.evaluation.external_benchmarks import load_encoder

    model = load_encoder(encoder, "cuda" if torch.cuda.is_available() else "cpu")
    sep = model.tokenizer.sep_token

    def encode(texts):
        vectors = model.encode(texts, batch_size=32).astype(np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    return encode, (lambda paper: f"{paper['title']}{sep}{paper['abstract']}")


def method_names(encoder):
    if encoder == "bge":  # names kept from the first run so saved results stay comparable
        return {"vec": "bge", "rrf": "rrf_bge_bm25", "full": "rrf_bge_bm25_cocitation"}
    return {"vec": encoder, "rrf": f"rrf_{encoder}_bm25", "full": f"rrf_{encoder}_bm25_cocitation"}


def run(dataset, split, encoder="bge"):
    from src.evaluation.citation_recommendation import cocitation_ranking, rrf
    from src.evaluation.retrievers.bm25_retrieval import BM25Retriever

    papers = load(dataset)
    train = set(split_ids(papers, dataset, "train"))
    queries = [(pid, gold(papers, pid, train)) for pid in split_ids(papers, dataset, split)]
    queries = [(pid, g) for pid, g in queries if len(g) >= MIN_GOLD]
    train_ids = sorted(train)
    print(f"{dataset}/{split}: {len(papers)} papers, {len(train_ids)} candidates, {len(queries)} queries", flush=True)

    cache = DATA / dataset / f"train_{encoder}.npy"  # derived from the downloaded corpus; not committed
    encode, doc_text = make_encoder(encoder)
    if cache.exists():
        doc = np.load(cache)
    else:
        doc = encode([doc_text(papers[p]) for p in train_ids])
        np.save(cache, doc)
    qvec = encode([doc_text(papers[p]) for p, _ in queries])
    names = method_names(encoder)
    bm25 = BM25Retriever([{"id": p, "title": papers[p]["title"], "abstract": papers[p]["abstract"]} for p in train_ids])
    # BM25 rankings do not depend on the encoder; cache them so each split is searched once.
    lexical_path = DATA / dataset / f"bm25_{split}.json"
    lexical_cache = json.loads(lexical_path.read_text()) if lexical_path.exists() else {}
    cites = {p: papers[p]["out"] & train for p in train_ids}

    per_query = {name: {} for name in ("bm25", names["vec"], names["rrf"], names["full"])}
    for i, (pid, gold_set) in enumerate(queries):
        scores = doc @ qvec[i]
        order = np.argsort(-scores)[:max(DEPTH, NEIGHBOURS)]
        bge = [train_ids[j] for j in order[:DEPTH]]
        if pid not in lexical_cache:
            lexical_cache[pid] = [r["id"] for r in bm25.retrieve(text(papers[pid]), top_k=DEPTH)]
        lexical = lexical_cache[pid]
        neighbours = [train_ids[j] for j in order[:NEIGHBOURS]]
        graph = cocitation_ranking(neighbours, [scores[j] for j in order[:NEIGHBOURS]], cites, lambda c: True)
        lists = {"bm25": lexical, names["vec"]: bge, names["rrf"]: rrf([bge, lexical], [1.0, 1.0]),
                 names["full"]: rrf([bge, lexical, graph], [1.0, 1.0, GRAPH_WEIGHT])}
        for name, ranked in lists.items():
            per_query[name][pid] = metrics(ranked[:DEPTH], gold_set)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(queries)}", flush=True)
    lexical_path.write_text(json.dumps(lexical_cache))
    return per_query


def main(argv=None) -> int:
    from src.evaluation.significance import paired_permutation_test

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(SPLITS), required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--encoder", choices=ENCODERS, default="bge")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    per_query = run(args.dataset, args.split, args.encoder)
    names = method_names(args.encoder)
    summary = {name: summarise(q) for name, q in per_query.items()}
    comparisons = {}
    for a, b in ((names["full"], names["rrf"]), (names["full"], names["vec"]), (names["rrf"], "bm25")):
        for metric in ("r@20", "rr"):
            deltas = [per_query[a][q][metric] - per_query[b][q][metric] for q in per_query[a]]
            comparisons[f"{a}_vs_{b}_{metric}"] = paired_permutation_test(deltas)
    report = {"dataset": args.dataset, "split": args.split, "encoder": args.encoder,
              "protocol": "citeomatic (Bhagavatula et al. 2018)",
              "min_gold": MIN_GOLD, "neighbours": NEIGHBOURS, "graph_weight": GRAPH_WEIGHT, "depth": DEPTH,
              "methods": summary, "comparisons": comparisons, "published": PUBLISHED[args.dataset]}
    suffix = "" if args.encoder == "bge" else f"_{args.encoder}"
    (OUT / f"{args.dataset}_{args.split}{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, s in summary.items():
        print(f"  {name:26} F1@20={s['f1@20']:.3f} MRR={s['mrr']:.3f} R@20={s['r@20']:.3f} (n={s['queries']})")
    for name, (f1, mrr) in PUBLISHED[args.dataset].items():
        print(f"  published {name:18} F1@20={f1:.3f} MRR={mrr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
