"""Query gate: generic / non-scientific inputs are refused before any mode runs."""
from src.pipeline.query_gate import candidate_terms, content_words, gate_decision

HIGH = [0.80, 0.78, 0.77, 0.76, 0.75]       # a real research question
BORDER = [0.64, 0.62, 0.61, 0.60, 0.59]     # short topic name, mean 0.612
LOW = [0.62, 0.60, 0.60, 0.59, 0.59]        # everyday sentence, mean 0.60


def test_greetings_and_fillers_have_no_content_words():
    assert content_words("hello how are you") == []
    assert content_words("hi!! ok thanks") == []
    assert content_words("what is 2026") == []
    assert content_words("graph neural networks for fraud detection") == ["graph", "neural", "networks", "fraud", "detection"]


def test_no_content_words_is_invalid_even_if_close():
    decision = gate_decision("hello how are you", HIGH)
    assert not decision["valid"] and decision["reason"] == "no_content_words"
    assert "Not a valid query" in decision["message"]


def test_close_to_corpus_is_valid():
    decision = gate_decision("graph neural networks for fraud detection", HIGH)
    assert decision["valid"] and decision["reason"] == "ok" and decision["message"] == ""


def test_borderline_needs_a_recognised_technical_term():
    assert gate_decision("operating systems", BORDER, ["operating systems"])["reason"] == "ok_scientific_term"
    rejected = gate_decision("funny memes", BORDER, [])
    assert not rejected["valid"] and rejected["reason"] == "off_topic"


def test_far_from_corpus_is_invalid_even_with_a_term():
    assert not gate_decision("weather forecast", [0.55, 0.54, 0.53, 0.52, 0.51], ["weather forecast"])["valid"]


def test_candidate_terms_skip_stop_word_edges():
    terms = candidate_terms("what is the best operating systems for embedded devices")
    assert "operating systems" in terms and "embedded devices" in terms
    assert not any(t.split()[0] in {"what", "is", "the", "for"} or t.split()[-1] in {"the", "for"} for t in terms)
