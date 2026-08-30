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
from src.pipeline.graph_only_retrieval import (
    graph_only_retrieve,
    normalize_query_concepts,
)


class FakeGraphSession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def run(self, query, **parameters):
        self.driver.calls.append(
            {
                "query": query,
                "parameters": parameters,
            }
        )
        return list(self.driver.records)


class FakeGraphDriver:
    def __init__(self, records):
        self.records = records
        self.calls = []
        self.session_count = 0

    def session(self):
        self.session_count += 1
        return FakeGraphSession(self)


def graph_records():
    return [
        {
            "id": "paper-1",
            "title": "Graph neural networks for nodes",
            "abstract": "Node classification using graph learning.",
            "year": 2024,
            "category": "Computer Science",
            "matchedConcepts": [
                "graph neural networks",
                "node classification",
            ],
            "matchedTerms": [
                "graph neural networks",
                "node classification",
            ],
            "citationDegree": 5,
        },
        {
            "id": "paper-2",
            "title": "General graph learning",
            "abstract": "Learning representations for graphs.",
            "year": 2023,
            "category": "Computer Science",
            "matchedConcepts": [
                "graph neural networks",
            ],
            "matchedTerms": [
                "graph neural networks",
            ],
            "citationDegree": 20,
        },
    ]


def test_query_concepts_are_normalized():
    concepts = normalize_query_concepts(
        "Graph Neural Networks for Node Classification"
    )

    assert "graph neural networks" in concepts
    assert "node classification" in concepts
    assert "for" not in concepts
    assert concepts == normalize_query_concepts(
        "Graph Neural Networks for Node Classification"
    )


def test_graph_only_uses_query_terms_not_chroma_seeds():
    driver = FakeGraphDriver(graph_records())

    graph_only_retrieve(
        driver,
        "graph neural networks for node classification",
        top_k=2,
    )

    assert len(driver.calls) == 1
    parameters = driver.calls[0]["parameters"]
    assert "query_terms" in parameters
    assert "paper_ids" not in parameters


def test_graph_only_is_deterministic_and_obeys_schema():
    driver = FakeGraphDriver(graph_records())

    first = graph_only_retrieve(
        driver,
        "graph neural networks for node classification",
        top_k=2,
    )
    second = graph_only_retrieve(
        driver,
        "graph neural networks for node classification",
        top_k=2,
    )

    assert first == second
    assert len(first) == 2
    assert first[0]["id"] == "paper-1"

    required_fields = {
        "id",
        "title",
        "abstract",
        "year",
        "category",
        "score",
        "source",
    }
    assert required_fields <= set(first[0])
    assert all(paper["source"] == "graph" for paper in first)


def test_graph_only_handles_no_concept_match():
    driver = FakeGraphDriver([])

    results = graph_only_retrieve(
        driver,
        "unknown scientific concept",
        top_k=10,
    )

    assert results == []


def test_graph_only_empty_query_does_not_open_neo4j():
    driver = FakeGraphDriver(graph_records())

    results = graph_only_retrieve(
        driver,
        "the and for",
        top_k=10,
    )

    assert results == []
    assert driver.session_count == 0


def test_graph_only_removes_duplicate_paper_ids():
    records = graph_records()
    records.append(dict(records[0]))
    driver = FakeGraphDriver(records)

    results = graph_only_retrieve(
        driver,
        "graph neural networks",
        top_k=10,
    )

    ids = [paper["id"] for paper in results]
    assert ids.count("paper-1") == 1


def test_graph_only_rejects_invalid_candidate_limit():
    driver = FakeGraphDriver(graph_records())

    with pytest.raises(ValueError):
        graph_only_retrieve(
            driver,
            "graph neural networks",
            candidate_limit=0,
        )


def test_graph_only_requires_neo4j_driver():
    with pytest.raises(ValueError):
        graph_only_retrieve(
            None,
            "graph neural networks",
        )