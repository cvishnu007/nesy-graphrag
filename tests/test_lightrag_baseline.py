import numpy as np

from src.evaluation.lightrag_baseline import LightRAGStyle, parse_keywords


def toy():
    """4 papers, entities 0..3; relation (0,1) w=2 and (1,2) w=2; unit vectors for matching."""
    m = LightRAGStyle.__new__(LightRAGStyle)
    m.papers = ["p0", "p1", "p2", "p3"]
    m.entities = ["gnn", "fraud", "graph", "vision"]
    m.paper_entities = [{0, 1}, {0, 1, 2}, {1, 2}, {3}]
    m.relations = [[0, 1, 2], [1, 2, 2]]
    eye = np.eye(4, dtype=np.float16)
    m.entity_vectors = eye
    m.relation_vectors = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.float16) / np.sqrt(2)
    m.entity_papers = {0: [0, 1], 1: [0, 1, 2], 2: [1, 2], 3: [3]}
    m.neighbours = {0: {1}, 1: {0, 2}, 2: {1}}
    m.degree = {0: 1, 1: 2, 2: 1}
    return m


def test_local_orders_by_entity_rank_then_neighbour_overlap(monkeypatch):
    m = toy()
    monkeypatch.setattr(LightRAGStyle, "_vector", lambda self, text: np.array([0, 1, 0.5, 0.1], dtype=np.float32))
    monkeypatch.setattr("src.evaluation.lightrag_baseline.TOP_K", 2)
    # top entities: fraud (1.0) then graph (0.5); fraud's papers sorted by overlap with {gnn, graph}: p1 (2) first
    assert m.local("fraud") == ["p1", "p0", "p2"]


def test_global_resorts_relations_by_degree_and_returns_shared_papers(monkeypatch):
    m = toy()
    monkeypatch.setattr(LightRAGStyle, "_vector", lambda self, text: np.array([1, 1, 0, 0], dtype=np.float32))
    monkeypatch.setattr("src.evaluation.lightrag_baseline.TOP_K", 1)
    assert m.global_("gnn fraud") == ["p0", "p1"]


def test_round_robin_dedupes_and_alternates():
    assert LightRAGStyle.round_robin(["a", "b", "c"], ["b", "d"], depth=4) == ["a", "b", "d", "c"]


def test_parse_keywords_reads_json_inside_text():
    out = parse_keywords('Sure: {"high_level_keywords": ["A", " "], "low_level_keywords": ["x", "y"]} done')
    assert out == {"high_level_keywords": ["A"], "low_level_keywords": ["x", "y"]}
