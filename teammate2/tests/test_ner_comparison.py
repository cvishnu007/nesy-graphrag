from src.evaluation.ner_comparison import (
    compare_ner_extractors,
    extract_scientific_pattern_concepts,
)


def records():
    return [
        {"id": "B", "clean_abstract": "Graph neural networks use message passing."},
        {"id": "A", "clean_abstract": "LightGCN improves collaborative filtering."},
        {"id": "C", "clean_abstract": "No technical phrase is present."},
    ]


def test_ner_comparison_uses_identical_deterministic_document_ids():
    baseline = lambda text: ["baseline"] if text else []
    alternative = lambda text: ["alternative"] if text else []

    first = compare_ner_extractors(
        records(), baseline, alternative, sample_size=2, seed="ner-v1"
    )
    second = compare_ner_extractors(
        list(reversed(records())), baseline, alternative,
        sample_size=2, seed="ner-v1",
    )

    assert first["document_ids"] == second["document_ids"]
    assert first["baseline"]["document_ids"] == first["alternative"]["document_ids"]
    assert first["controlled_variables"]["only_changed_component"] == "concept_extractor"


def test_scientific_pattern_extractor_is_deterministic_and_finds_terms():
    text = "LightGCN is a graph neural network for collaborative filtering."

    first = extract_scientific_pattern_concepts(text)
    second = extract_scientific_pattern_concepts(text)

    assert first == second
    assert "lightgcn" in first
    assert "graph neural network" in first
    assert "collaborative filtering" in first


def test_scientific_model_tokens_do_not_match_gat_inside_ordinary_words():
    concepts = extract_scientific_pattern_concepts(
        "We investigate aggregation and propagation behavior."
    )

    assert "investigate" not in concepts
    assert "aggregation" not in concepts
    assert "propagation" not in concepts
