import pytest

from src.evaluation.retrievers.graph_only_retrieval import (
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
            "matchedConcepts": ["graph neural networks"],
            "matchedTerms": ["graph neural networks"],
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
    assert graph_only_retrieve(
        driver,
        "unknown scientific concept",
        top_k=10,
    ) == []


def test_graph_only_empty_query_does_not_open_neo4j():
    driver = FakeGraphDriver(graph_records())
    assert graph_only_retrieve(driver, "the and for", top_k=10) == []
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
        graph_only_retrieve(None, "graph neural networks")
