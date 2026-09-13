"""Controlled vector-RAG/GraphRAG generation with identical context/generator limits.

The budget is explicitly measured in formatted Unicode characters, not tokens.
Both arms verify paper identity against the corpus, so the vector arm does not
need a graph validation step. Passage provenance measures traceability, not truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from src.pipeline.prompts import build_review_prompt
from src.pipeline.provenance import (
    build_passages, format_passage_context, parse_review_claims,
    validate_claim_provenance, render_grounded_review,
)


def bounded_passages(papers, corpus, *, max_chars):
    canonical = []
    seen = set()
    for paper in papers:
        pid = paper['id']
        if pid not in corpus or pid in seen:
            continue
        seen.add(pid)
        item = corpus[pid]
        canonical.append({**item, 'title': item.get('clean_title', item.get('title', '')),
                          'abstract': item.get('clean_abstract', item.get('abstract', ''))})
    passages = build_passages(canonical, {p['id']: p['title'] for p in canonical})
    selected = []
    for passage in passages:
        if len(format_passage_context(selected + [passage])) > max_chars:
            break
        selected.append(passage)
    return selected


def compare_reviews(query, retrievers, corpus, client, *, top_k=5, context_chars=6000, max_tokens=600):
    for value in [top_k, context_chars, max_tokens]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError('Retrieval, context and output budgets must be positive integers')
    rows = []
    for method, retrieve in retrievers.items():
        start = perf_counter()
        papers = retrieve(query, top_k)
        passages = bounded_passages(papers, corpus, max_chars=context_chars)
        context = format_passage_context(passages)
        prompt = build_review_prompt(context, query)
        raw, error = '', ''
        if passages:
            try:
                raw = client.complete(prompt, max_tokens=max_tokens, temperature=0.0)
            except Exception as exc:
                error = type(exc).__name__
        else:
            error = 'no_evidence_within_budget'
        claims, errors = parse_review_claims(raw)
        provenance = validate_claim_provenance(claims, passages, parse_errors=errors)
        rows.append({
            'method': method, 'query': query, 'paper_ids': [p['id'] for p in papers],
            'context_paper_ids': sorted({p['paper_id'] for p in passages}),
            'passages': passages, 'context_chars': len(context),
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'raw_output': raw, 'provenance': provenance,
            'answer': render_grounded_review(provenance['claims']),
            'error': error, 'latency_seconds': perf_counter() - start,
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', default='evaluation/benchmarks/retrieval_queries.json')
    parser.add_argument('--split', choices=['dev', 'test'], default='test')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--context-chars', type=int, default=6000)
    parser.add_argument('--max-tokens', type=int, default=600)
    parser.add_argument('--query-limit', type=int, help='Smoke sample only; never describe as full benchmark')
    args = parser.parse_args(argv)
    from src.evaluation.benchmark_io import load_benchmark
    from src.pipeline.retrieval import vector_only_retrieve, nesy_retrieve
    from src.storage.neo4j_store import get_driver
    from src.utils.config import CLEAN_FILE
    from src.utils.llm_client import create_llm_client, llm_runtime_metadata
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh directory; comparison outputs are immutable')
    benchmark = load_benchmark(args.benchmark)
    queries = [q for q in benchmark['queries'] if q['split'] == args.split]
    if args.query_limit is not None:
        if args.query_limit <= 0:
            raise ValueError('query-limit must be positive')
        queries = queries[:args.query_limit]
    corpus_bytes = Path(CLEAN_FILE).read_bytes()
    corpus = {p['id']: p for p in json.loads(corpus_bytes)}
    from src.evaluation.refresh_retrieval import runtime_retrieval_settings
    protocol = {
        'benchmark_version': benchmark['benchmark_version'], 'split': args.split,
        'retrieval_settings': runtime_retrieval_settings(),
        'query_ids': [q['query_id'] for q in queries], 'sample_only': args.query_limit is not None,
        'corpus_sha256': hashlib.sha256(corpus_bytes).hexdigest(), 'paper_count': len(corpus),
        'llm': llm_runtime_metadata(), 'top_k': args.top_k, 'context_chars': args.context_chars,
        'max_tokens': args.max_tokens, 'temperature': 0.0, 'repair_attempts': 0,
        'selection': 'predeclared defaults; no test-set tuning',
        'interpretation': 'structural provenance and latency only; semantic correctness requires separate labels',
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    driver = get_driver()
    try:
        client = create_llm_client()
        retrievers = {'vector_rag': lambda q, k: vector_only_retrieve(q, top_k=k),
                      'graph_rag': lambda q, k: nesy_retrieve(driver, q, top_k=k)}
        with (output / 'reviews.jsonl').open('w') as stream:
            for query in queries:
                for row in compare_reviews(query['query'], retrievers, corpus, client,
                        top_k=args.top_k, context_chars=args.context_chars, max_tokens=args.max_tokens):
                    stream.write(json.dumps({**row, 'query_id': query['query_id'], 'split': args.split}) + '\n')
                    stream.flush()
                print(f"Completed {query['query_id']}", flush=True)
    finally:
        driver.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
