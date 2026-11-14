from src.pipeline.contradiction_citation import CRITICAL, rank_citation_candidates, shared_subjects


def test_critical_language_detected():
    assert CRITICAL.search("We show that graph neural networks fail to outperform greedy heuristics.")
    assert CRITICAL.search("Contrary to prior work, attention does not improve accuracy.")
    assert not CRITICAL.search("Our method achieves state-of-the-art accuracy on three datasets.")


def test_subject_must_appear_in_both_sentences():
    entities = {"graph neural networks", "greedy heuristics", "mis"}
    assert shared_subjects("graph neural networks fail here", "graph neural networks solve it", entities) == \
        ["graph neural networks"]
    assert shared_subjects("transformers fail", "graph neural networks win", entities) == []


def test_rank_pairs_critical_citing_sentence_with_cited_paper():
    citing = {"id": "c", "title": "Revisiting", "year": 2023,
              "abstract": "We show that graph neural networks fail to outperform simple greedy heuristics on large instances."}
    cited = {"id": "d", "title": "GNN solver", "year": 2021,
             "abstract": "Our experiments show graph neural networks outperform greedy heuristics on large instances."}
    out = rank_citation_candidates([(citing, cited, {"graph neural networks", "greedy heuristics"})],
                                   scorer=lambda pairs: [0.9] * len(pairs))
    assert out and out[0]["paper1"]["id"] == "c" and "graph neural networks" in out[0]["shared_subjects"]
    none = rank_citation_candidates([(cited, citing, {"graph neural networks"})], scorer=lambda p: [0.9] * len(p))
    assert none == []  # the citing paper here has no critical sentence
