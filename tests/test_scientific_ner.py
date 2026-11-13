import json

from src.ingestion.scientific_ner import (
    bio_to_spans, extract_corpus, load_scierc, normalize_entity, score_predictions, word_tokenize,
)


def test_bio_decoding_handles_boundaries_and_stray_inside_tags():
    labels = ["B-Method", "I-Method", "O", "I-Task", "I-Task", "B-Metric", "B-Metric"]
    assert bio_to_spans(labels) == [(0, 1, "Method"), (3, 4, "Task"), (5, 5, "Metric"), (6, 6, "Metric")]
    assert bio_to_spans(["B-Task", "I-Method"]) == [(0, 0, "Task"), (1, 1, "Method")]


def test_scierc_loader_converts_document_offsets_to_sentence_offsets(tmp_path):
    path = tmp_path / "test.json"
    path.write_text(json.dumps({"doc_key": "D", "sentences": [["a", "b"], ["c", "d", "e"]],
                                "ner": [[[0, 1, "Task"]], [[3, 4, "Method"]]]}) + "\n")
    sentences = load_scierc(path)
    assert sentences[1]["gold"] == {(1, 2, "Method")}


def test_typed_and_untyped_scores():
    sentences = [{"gold": {(0, 1, "Task"), (3, 3, "Method")}}]
    predictions = [[(0, 1, "Method"), (3, 3, "Method")]]
    assert score_predictions(sentences, predictions, typed=False)["f1"] == 1.0
    typed = score_predictions(sentences, predictions, typed=True)
    assert typed["precision"] == 0.5 and typed["per_type"]["Method"]["recall"] == 1.0


def test_extraction_drops_generic_and_deduplicates():
    class Fake:
        def predict(self, batches):
            return [[(0, 1, "Method"), (3, 3, "Generic")] for _ in batches]

    papers = [{"id": "p", "abstract": "Graph Transformers beat our method. Graph transformers again win."}]
    records = extract_corpus(Fake(), papers)
    assert records == [{"id": "p", "entities": [{"text": "graph transformers", "type": "Method"}]}]


def test_tokenizer_and_normalizer():
    assert word_tokenize("BERT-based models (e.g., GNNs) work.") == ["BERT-based", "models", "(", "e", ".", "g", ".", ",", "GNNs", ")", "work", "."]
    assert normalize_entity("  Graph  Neural Networks. ") == "graph neural networks"


def test_concept_cypher_switches_graph_label_and_relationship():
    from src.pipeline.concepts import concept_cypher

    query = "MATCH (p)-[:RELATED_TO]->(c:Concept) MATCH path=(a)-[:RELATED_TO*..6]-(b)"
    assert concept_cypher(query, "spacy") == query
    assert concept_cypher(query, "scientific") == (
        "MATCH (p)-[:MENTIONS]->(c:SciEntity) MATCH path=(a)-[:MENTIONS*..6]-(b)")


def test_flat_bio_keeps_longest_of_nested_spans():
    from src.ingestion.train_scierc_ner import flat_bio

    tokens = ["recognition", "of", "proper", "nouns", "in", "text"]
    labels = flat_bio(tokens, {(0, 3, "Task"), (2, 3, "OtherScientificTerm"), (5, 5, "Material")})
    assert labels == ["B-Task", "I-Task", "I-Task", "I-Task", "O", "B-Material"]


def test_scores_can_exclude_generic_type():
    sentences = [{"gold": {(0, 0, "Generic"), (2, 3, "Method")}}]
    predictions = [[(2, 3, "Method")]]
    assert score_predictions(sentences, predictions, typed=True)["recall"] == 0.5
    without = score_predictions(sentences, predictions, typed=True, exclude_types=("Generic",))
    assert without["f1"] == 1.0 and "Generic" not in without["per_type"]


def test_normalizer_drops_abbreviation_fragments():
    assert normalize_entity("convolutional neural networks ( cnns") == "convolutional neural networks"
    assert normalize_entity("graph neural networks ( gnns )") == "graph neural networks"
    assert normalize_entity("state - of - the - art") == "state-of-the-art"
    assert normalize_entity("long short-term memory ( lstm ) networks") == "long short-term memory (lstm) networks"
