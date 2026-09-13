import math

import pytest

from src.evaluation.retrievers.bm25_retrieval import BM25Retriever


def test_bm25_ranks_exact_terms_and_has_expected_score():
    retriever = BM25Retriever([
        {'id': 'b', 'clean_title': 'graph graph', 'clean_abstract': ''},
        {'id': 'a', 'clean_title': 'database query', 'clean_abstract': ''},
    ])
    rows = retriever.retrieve('graph', 20)
    assert [r['id'] for r in rows] == ['b']
    assert rows[0]['score'] == pytest.approx(math.log(2) * 2 * 2.5 / 3.5)
    assert rows[0]['source'] == 'bm25'


def test_bm25_order_is_independent_of_input_order_and_empty_query_returns_none():
    papers = [{'id': p, 'title': 'Graph networks', 'abstract': 'learning'} for p in ['b', 'a']]
    assert [r['id'] for r in BM25Retriever(papers).retrieve('GRAPH', 1)] == ['a']
    assert BM25Retriever(papers).retrieve('unknown', 20) == []
    assert BM25Retriever(papers).retrieve('', 20) == []


def test_bm25_rejects_duplicate_ids_and_invalid_parameters():
    with pytest.raises(ValueError, match='duplicate'):
        BM25Retriever([{'id': 'a'}, {'id': 'a'}])
    with pytest.raises(ValueError):
        BM25Retriever([], k1=0)
    with pytest.raises(ValueError):
        BM25Retriever([]).retrieve('graph', 0)


def test_pool_and_evaluation_include_optional_baselines():
    from src.evaluation.candidate_pool import pool_query
    from src.evaluation.retrieval_runner import evaluate_benchmark
    paper = {'id': 'p1', 'title': 'graph', 'abstract': 'graph', 'year': 2020,
             'category': 'cs', 'score': 1., 'source': 'fixture'}
    retrievers = {name: lambda q, k: [paper] for name in ['vector', 'graph', 'hybrid', 'bm25', 'production']}
    query = {'query_id': 'q1', 'query': 'graph', 'split': 'dev', 'judgments': {'p1': 2}}
    hidden, audit, summary = pool_query(query, retrievers, benchmark_version='test', pool_depth=20)
    assert set(summary['method_counts']) == set(retrievers)
    assert set(audit[0]['methods']) == set(retrievers)
    assert 'methods' not in hidden[0]
    benchmark = {'benchmark_version': 'test', 'queries': [query]}
    rows, _, summary = evaluate_benchmark(benchmark, retrievers, split='dev')
    assert set(summary['methods']) == set(retrievers)
    assert len(rows) == 5
