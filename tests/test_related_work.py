from src.pipeline.related_work import fuse


def test_fuse_combines_sources_and_excludes():
    out = fuse(["a", "b"], ["b", "c"], ["c", "b"], top_k=3, exclude={"a"})
    ids = [pid for pid, _, _ in out]
    assert ids[0] == "b" and "a" not in ids
    assert out[0][2] == {"bge": 2, "bm25": 1, "cocitation": 2}


def test_graph_weight_zero_ignores_cocitation_order():
    out = fuse(["x"], ["y"], ["z"], top_k=3, graph_weight=0.0)
    assert [pid for pid, _, _ in out][-1] == "z"


class _Session:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, *args, **kwargs):
        return self.rows


class _Driver:
    def __init__(self, rows):
        self.rows = rows

    def session(self):
        return _Session(self.rows)


def test_citation_graph_keeps_top_neighbours_and_inner_citations():
    from src.pipeline.related_work import citation_graph

    results = [{"id": "s1"}, {"id": "s2"}]
    neighbours = [{"id": n, "title": n.upper(), "year": 2021} for n in ("n1", "n2", "n3")]
    citers = {"s1": ["n1", "n2"], "s2": ["n1", "s1"]}
    graph = citation_graph(_Driver([{"a": "s1", "b": "s2"}]), results, neighbours, citers, max_neighbours=1)
    assert [n["id"] for n in graph["nodes"]] == ["n1"]
    assert graph["nodes"][0]["cites_suggestions"] == 2
    kinds = sorted((e["source"], e["target"], e["kind"]) for e in graph["edges"])
    assert kinds == [("n1", "s1", "cocitation"), ("n1", "s2", "cocitation"), ("s1", "s2", "cites")]


def test_graph_view_numbers_suggestions_and_links_query():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    from graph_view import build_graph, graph_html

    result = {"query": "q", "papers": [
        {"id": "s1", "title": "T1", "year": 2020, "why": "w", "source_ranks": {"bge": 1, "cocitation": 3}},
        {"id": "s2", "title": "T2", "year": 2021, "why": "w", "source_ranks": {"bm25": 2}}],
        "graph": {"nodes": [{"id": "n1", "title": "N", "year": 2019, "cites_suggestions": 1}],
                  "edges": [{"source": "n1", "target": "s1", "kind": "cocitation"}]}}
    nodes, edges, colors = build_graph(result, theme="light")
    labels = {n["id"]: n["label"] for n in nodes}
    assert labels["s1"] == "1" and labels["s2"] == "2"
    assert nodes[1]["color"] == colors["graph"] and nodes[2]["color"] == colors["text_only"]
    assert {(e["from"], e["to"]) for e in edges} == {("__query__", "s1"), ("__query__", "s2"), ("n1", "s1")}
    page = graph_html(result)
    # The bundled library is inlined so the map works offline; no CDN request.
    assert "@version 9.1.9" in page and "unpkg.com" not in page


def test_attach_citation_evidence_titles_in_similarity_order():
    from src.pipeline.related_work import attach_citation_evidence

    results = [{"id": "s1", "title": "S1", "year": 2020}, {"id": "s2", "title": "S2", "year": 2021}]
    neighbours = [{"id": "n2", "title": "N2", "year": 2022}, {"id": "n1", "title": "N1", "year": 2023}]
    graph = {"edges": [{"source": "s1", "target": "s2", "kind": "cites"}]}
    attach_citation_evidence(results, neighbours, {"s1": ["n1", "n2"]}, graph)
    assert [x["title"] for x in results[0]["cited_by_similar"]] == ["N2", "N1"]
    assert results[0]["cites"] == [{"id": "s2", "title": "S2", "year": 2021}]
    assert results[1]["cited_by"][0]["id"] == "s1" and results[1]["cited_by_similar"] == []
