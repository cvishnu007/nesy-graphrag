import pandas as pd

from src.ingestion import ner_extractor
from src.ingestion.semantic_scholar_fetcher import (
    fetch_seed_papers,
    merge_paper_records,
    normalize_papers,
)


class SinglePageClient:
    def __init__(self):
        self.params = None

    def request(self, method, path, params=None, json_body=None):
        self.params = params
        return {
            "data": [
                {"paperId": "paper-1", "title": "Paper one"},
                {"paperId": "paper-2", "title": "Paper two"},
            ]
        }


def test_seed_fetch_uses_explicit_topic_and_limit():
    client = SinglePageClient()

    papers = fetch_seed_papers(client, query="cybersecurity", limit=2)

    assert [paper["paperId"] for paper in papers] == ["paper-1", "paper-2"]
    assert client.params["query"] == "cybersecurity"
    assert client.params["limit"] >= 2


def test_cross_topic_merge_preserves_references_and_provenance():
    existing = {
        "paper-1": {
            "id": "paper-1",
            "paperId": "paper-1",
            "references": ["ref-a"],
            "categories": ["Computer Science"],
            "ingestion_queries": ["graph neural networks"],
            "citationCount": 2,
            "referenceCount": 1,
        }
    }
    incoming = [{
        "id": "paper-1",
        "paperId": "paper-1",
        "references": ["ref-a", "ref-b"],
        "categories": ["Computer Science", "Engineering"],
        "ingestion_queries": ["artificial intelligence and machine learning"],
        "citationCount": 5,
        "referenceCount": 2,
    }]

    merged = merge_paper_records(existing, incoming)["paper-1"]

    assert merged["references"] == ["ref-a", "ref-b"]
    assert merged["ingestion_queries"] == [
        "graph neural networks",
        "artificial intelligence and machine learning",
    ]
    assert merged["citationCount"] == 5
    assert merged["referenceCount"] == 2


def test_normalization_records_ingestion_topic():
    normalized = normalize_papers(
        [{"paperId": "paper-1", "title": "Title", "abstract": "Abstract"}],
        {"paper-1": ["ref-a"]},
        ingestion_query="computer vision",
    )

    assert normalized[0]["ingestion_queries"] == ["computer vision"]
    assert normalized[0]["references"] == ["ref-a"]


def test_ner_resume_reuses_complete_existing_output(tmp_path, monkeypatch):
    clean_path = tmp_path / "clean.json"
    ner_path = tmp_path / "ner.json"
    clean = pd.DataFrame([
        {"id": "paper-1", "clean_abstract": "first abstract"},
        {"id": "paper-2", "clean_abstract": "second abstract"},
    ])
    existing = clean.copy()
    existing["entities"] = [["first entity"], ["second entity"]]
    clean.to_json(clean_path, orient="records")
    existing.to_json(ner_path, orient="records")

    monkeypatch.setattr(ner_extractor, "CLEAN_FILE", str(clean_path))
    monkeypatch.setattr(ner_extractor, "NER_FILE", str(ner_path))
    monkeypatch.setattr(
        ner_extractor.spacy,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("spaCy should not load when all papers are checkpointed")
        ),
    )

    ner_extractor.run()

    result = pd.read_json(ner_path)
    assert result["entities"].tolist() == [["first entity"], ["second entity"]]
