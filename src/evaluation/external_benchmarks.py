"""Retrieval evaluation on public scientific benchmarks with external labels.

Benchmarks (downloaded to data/external/, not redistributed):

* SciDocs (BEIR version; Cohan et al., 2020; Thakur et al., 2021): 1,000
  citation-prediction queries (paper titles) over 25,657 papers. Relevance is
  derived from citation and co-view behaviour, not graded by annotators.
* LitSearch (Ajith et al., EMNLP 2024): 597 literature-search questions over
  64,183 ACL/ICLR papers. Questions were generated from paper citations (GPT-4
  assisted) and manually inspected/edited by the authors, or written by
  researchers about their own papers. Relevance is the set of target papers.

Every method uses the documented input format of its model. Labels come only
from the benchmarks; nothing is graded by this project.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from src.evaluation.ir_metrics import aggregate_query_metrics, evaluate_ranking

ROOT = Path("data/external")
K_VALUES = (5, 10, 20, 100)

MODELS = {
    "specter": {"repo": "sentence-transformers/allenai-specter", "kind": "st", "doc": "sep"},
    "specter2_base": {"repo": "allenai/specter2_aug2023refresh_base", "kind": "cls", "doc": "sep"},
    "scincl": {"repo": "malteos/scincl", "kind": "cls", "doc": "sep"},
    # OpenScholar retriever (Asai et al., 2024): Contriever trained on scientific papers; mean pooling.
    "openscholar": {"repo": "OpenSciLM/OpenScholar_Retriever", "kind": "mean", "doc": "dot"},
    "minilm": {"repo": "sentence-transformers/all-MiniLM-L6-v2", "kind": "st", "doc": "dot"},
    "bge_base": {"repo": "BAAI/bge-base-en-v1.5", "kind": "st", "doc": "dot",
                 "query_prefix": "Represent this sentence for searching relevant passages: "},
}


# -- loaders ---------------------------------------------------------------

def _read_parquet(path, columns=None):
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns).to_pylist()


def load_scidocs(root=ROOT):
    base = Path(root) / "BeIR__scidocs"
    corpus = {str(row["_id"]): {"title": row.get("title") or "", "abstract": row.get("text") or ""}
              for row in _read_parquet(base / "corpus/corpus-00000-of-00001.parquet", ["_id", "title", "text"])}
    queries = {str(row["_id"]): row["text"]
               for row in _read_parquet(base / "queries/queries-00000-of-00001.parquet", ["_id", "text"])}
    qrels = defaultdict(dict)
    lines = (Path(root) / "BeIR__scidocs-qrels/test.tsv").read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        query_id, doc_id, score = line.split("\t")
        qrels[query_id][doc_id] = int(score)
    queries = {qid: text for qid, text in queries.items() if qid in qrels}
    return {"name": "scidocs", "corpus": corpus, "queries": queries, "qrels": dict(qrels),
            "label_source": "citation/co-view derived (BEIR SciDocs)"}


def load_litsearch(root=ROOT):
    base = Path(root) / "princeton-nlp__LitSearch"
    corpus = {}
    for path in sorted((base / "corpus_clean").glob("full-*.parquet")):
        for row in _read_parquet(path, ["corpusid", "title", "abstract"]):
            corpus[str(row["corpusid"])] = {"title": row.get("title") or "", "abstract": row.get("abstract") or ""}
    queries, qrels, meta = {}, {}, {}
    for index, row in enumerate(_read_parquet(base / "query/full-00000-of-00001.parquet")):
        query_id = f"LS{index:04d}"
        targets = [str(value) for value in row["corpusids"] if str(value) in corpus]
        if not targets:
            continue
        queries[query_id] = row["query"]
        qrels[query_id] = {doc: 1 for doc in targets}
        meta[query_id] = {key: row.get(key) for key in ("query_set", "specificity", "quality")}
    return {"name": "litsearch", "corpus": corpus, "queries": queries, "qrels": qrels, "query_meta": meta,
            "label_source": "author-inspected target papers (LitSearch)"}


# -- retrieval -------------------------------------------------------------

def document_text(paper, style, sep_token="[SEP]"):
    title, abstract = paper["title"].strip(), paper["abstract"].strip()
    if style == "sep":
        return f"{title}{sep_token}{abstract}"
    return ". ".join(part for part in (title, abstract) if part)


class ClsEncoder:
    """CLS- or mean-pooled Hugging Face encoder (SPECTER2 base, SciNCL, OpenScholar)."""

    def __init__(self, repo, device, pooling="cls"):
        self.pooling = pooling
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo).to(device).eval()
        if device == "cuda":
            self.model = self.model.half()

    def encode(self, texts, batch_size=32):
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                       max_length=512, return_tensors="pt").to(self.device)
                hidden = self.model(**batch).last_hidden_state
                if self.pooling == "mean":
                    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                else:
                    pooled = hidden[:, 0]
                outputs.append(pooled.float().cpu().numpy())
        return np.vstack(outputs)


def load_encoder(name, device):
    spec = MODELS[name]
    if spec["kind"] in {"cls", "mean"}:
        return ClsEncoder(spec["repo"], device, pooling=spec["kind"])
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(spec["repo"], device=device)
    if device == "cuda":
        model = model.half()
    model.max_seq_length = min(512, model.max_seq_length or 512)
    return model


def _normalize(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def dense_rankings(bench, name, *, device, depth=100, cache_dir=ROOT / "cache"):
    spec = MODELS[name]
    doc_ids = sorted(bench["corpus"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{bench['name']}_{name}.npy"
    encoder = None
    cached = cache.exists()
    started = perf_counter()
    if cached:
        doc_vectors = np.load(cache)
    else:
        encoder = load_encoder(name, device)
        sep = getattr(getattr(encoder, "tokenizer", None), "sep_token", None) or "[SEP]"
        texts = [document_text(bench["corpus"][doc], spec["doc"], sep) for doc in doc_ids]
        order = np.argsort([len(text) for text in texts])  # length-sorted batches are faster
        encoded = encoder.encode([texts[i] for i in order], batch_size=32)
        vectors = np.empty_like(encoded)
        vectors[order] = encoded
        doc_vectors = _normalize(vectors)
        np.save(cache, doc_vectors)
    encode_seconds = perf_counter() - started
    if encoder is None:
        encoder = load_encoder(name, device)
    query_ids = sorted(bench["queries"])
    prefix = spec.get("query_prefix", "")
    query_vectors = _normalize(encoder.encode([prefix + bench["queries"][q] for q in query_ids], batch_size=32))
    import torch

    docs = torch.tensor(doc_vectors, device=device)
    rankings = {}
    for start in range(0, len(query_ids), 256):
        scores = torch.tensor(query_vectors[start:start + 256], device=device) @ docs.T
        top = torch.topk(scores, min(depth, len(doc_ids)), dim=1).indices.cpu().numpy()
        for offset, row in enumerate(top):
            rankings[query_ids[start + offset]] = [doc_ids[i] for i in row]
    return rankings, {"corpus_encode_seconds": round(encode_seconds, 1), "cached": cached}


def bm25_rankings(bench, *, depth=100):
    from src.evaluation.retrievers.bm25_retrieval import BM25Retriever

    retriever = BM25Retriever([{"id": doc, **paper} for doc, paper in bench["corpus"].items()])
    return {qid: [hit["id"] for hit in retriever.retrieve(text, top_k=depth)]
            for qid, text in bench["queries"].items()}


def rrf(rankings_by_method, weights, *, depth=100, k=60):
    fused = {}
    for query_id in next(iter(rankings_by_method.values())):
        scores = defaultdict(float)
        for method, weight in weights.items():
            for rank, doc in enumerate(rankings_by_method[method][query_id], 1):
                scores[doc] += weight / (k + rank)
        fused[query_id] = sorted(scores, key=lambda doc: (-scores[doc], doc))[:depth]
    return fused


def score(bench, rankings):
    rows = [{"query_id": qid, **evaluate_ranking(rankings[qid], bench["qrels"][qid], K_VALUES)}
            for qid in sorted(bench["queries"])]
    return rows, aggregate_query_metrics(rows)["metrics"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark", choices=("scidocs", "litsearch"))
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fuse-with-bm25", nargs="*", default=None,
                        help="Dense models to fuse with BM25 by RRF (default: all requested)")
    args = parser.parse_args(argv)
    import torch
    from src.evaluation.significance import paired_permutation_test

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench = load_scidocs() if args.benchmark == "scidocs" else load_litsearch()
    print(f"{bench['name']}: {len(bench['corpus'])} papers, {len(bench['queries'])} queries", flush=True)

    rankings, timing = {"bm25": bm25_rankings(bench)}, {}
    for name in args.models:
        rankings[name], timing[name] = dense_rankings(bench, name, device=device)
        print(f"ranked {name} {timing[name]}", flush=True)
    fuse = args.models if args.fuse_with_bm25 is None else args.fuse_with_bm25
    for name in fuse:
        rankings[f"rrf_bm25_{name}"] = rrf({"bm25": rankings["bm25"], name: rankings[name]},
                                           {"bm25": 1.0, name: 1.0})

    per_query, summary = {}, {}
    for method, ranked in rankings.items():
        rows, metrics = score(bench, ranked)
        per_query[method] = {row["query_id"]: row for row in rows}
        summary[method] = {key: metrics[key]["mean"] for key in
                           ("ndcg@10", "recall@5", "recall@20", "recall@100", "map", "mrr")}

    reference = "specter"
    tests = {}
    for method in rankings:
        if method == reference:
            continue
        deltas = [per_query[method][q]["ndcg@10"] - per_query[reference][q]["ndcg@10"] for q in sorted(bench["queries"])]
        tests[f"{method}_vs_{reference}_ndcg@10"] = paired_permutation_test(deltas)
    subsets = {}
    if bench.get("query_meta"):
        groups = defaultdict(list)
        for qid, meta in bench["query_meta"].items():
            groups[f"set={meta.get('query_set')}"].append(qid)
            groups[f"specificity={meta.get('specificity')}"].append(qid)
        for group, qids in groups.items():
            subsets[group] = {"query_count": len(qids), **{
                method: {"recall@5": float(np.mean([per_query[method][q]["recall@5"] for q in qids])),
                         "recall@20": float(np.mean([per_query[method][q]["recall@20"] for q in qids]))}
                for method in rankings}}

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps({
        "benchmark": bench["name"], "label_source": bench["label_source"],
        "paper_count": len(bench["corpus"]), "query_count": len(bench["queries"]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "device": device,
        "models": {name: MODELS[name] for name in args.models}, "timing": timing,
        "methods": summary, "significance_vs_specter": tests, "subsets": subsets,
        "notes": "specter2_base is the SPECTER2 base model without its task adapter "
                 "(the adapters library would downgrade transformers in this environment).",
    }, indent=2) + "\n")
    with (output / "rankings.jsonl").open("w") as stream:
        for method, ranked in rankings.items():
            for qid in sorted(ranked):
                stream.write(json.dumps({"method": method, "query_id": qid, "ranked": ranked[qid][:100]}) + "\n")
    for method, values in sorted(summary.items(), key=lambda item: -item[1]["ndcg@10"]):
        print(f"{method:24} ndcg@10={values['ndcg@10']:.4f} R@5={values['recall@5']:.4f} "
              f"R@20={values['recall@20']:.4f} R@100={values['recall@100']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
