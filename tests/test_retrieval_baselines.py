import pandas as pd
import pytest

from src.pipeline.bm25_retrieval import (
    BM25Index,
    load_bm25_index,
    tokenize,
)


def sample_papers():
    return [
        {
            "id": "relevant",
            "title": "Graph neural networks for node classification",
            "abstract": (
                "A graph learning method for classifying nodes "
                "in citation networks."
            ),
            "year": 2024,
            "category": "Computer Science",
        },
        {
            "id": "partial",
            "title": "Graph representation learning",
            "abstract": "Learning representations for graph data.",
            "year": 2023,
            "category": "Computer Science",
        },
        {
            "id": "unrelated",
            "title": "Cooking recipes",
            "abstract": "Methods for baking bread and cakes.",
            "year": 2022,
            "category": "Other",
        },
    ]


def test_tokenize_is_lowercase_and_deterministic():
    assert tokenize("Graph Neural-Networks 2025!") == [
        "graph",
        "neural",
        "networks",
        "2025",
    ]


def test_bm25_ranks_exact_topic_first():
    index = BM25Index(sample_papers())

    results = index.search(
        "graph neural networks for node classification",
        top_k=3,
    )

    assert results[0]["id"] == "relevant"
    assert all(
        results[index]["score"] >= results[index + 1]["score"]
        for index in range(len(results) - 1)
    )


def test_bm25_is_deterministic_and_unique():
    index = BM25Index(sample_papers())

    first = index.search("graph learning", top_k=3)
    second = index.search("graph learning", top_k=3)

    assert first == second
    ids = [paper["id"] for paper in first]
    assert len(ids) == len(set(ids))


def test_bm25_obeys_top_k_and_common_schema():
    index = BM25Index(sample_papers())

    results = index.search("graph learning", top_k=1)

    assert len(results) == 1
    assert set(results[0]) == {
        "id",
        "title",
        "abstract",
        "year",
        "category",
        "score",
        "source",
    }
    assert results[0]["source"] == "bm25"


def test_empty_query_and_zero_top_k_return_empty():
    index = BM25Index(sample_papers())

    assert index.search("", top_k=10) == []
    assert index.search("graph", top_k=0) == []


def test_empty_corpus_returns_empty():
    index = BM25Index([])

    assert index.search("graph", top_k=10) == []


def test_duplicate_paper_ids_are_rejected():
    papers = sample_papers()
    papers.append(dict(papers[0]))

    with pytest.raises(ValueError):
        BM25Index(papers)


def test_invalid_k1_is_rejected():
    with pytest.raises(ValueError):
        BM25Index(sample_papers(), k1=0)


@pytest.mark.parametrize("invalid_b", [-0.1, 1.1])
def test_invalid_b_is_rejected(invalid_b):
    with pytest.raises(ValueError):
        BM25Index(sample_papers(), b=invalid_b)


def test_missing_clean_corpus_columns_are_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    pd.DataFrame([{"id": "paper-1"}]).to_json(
        path,
        orient="records",
    )

    with pytest.raises(RuntimeError):
        load_bm25_index(path)


def test_load_bm25_index_from_clean_file(tmp_path):
    path = tmp_path / "clean.json"
    dataframe = pd.DataFrame(
        [
            {
                "id": "paper-1",
                "clean_title": "graph node classification",
                "clean_abstract": "graph neural network method",
                "year": 2024,
                "primary_category": "Computer Science",
            }
        ]
    )
    dataframe.to_json(
        path,
        orient="records",
    )

    index = load_bm25_index(path)
    results = index.search("node classification", top_k=5)

    assert len(results) == 1
    assert results[0]["id"] == "paper-1"