from src.pipeline.retrieval import (
    build_retrieval_diagnostics,
    fuse_results,
    symbolic_expand,
)


class EmptyGraphSession:
    def __init__(self):
        self.query = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def run(self, query, **parameters):
        self.query = query
        return []


class EmptyGraphDriver:
    def __init__(self):
        self.active_session = EmptyGraphSession()

    def session(self):
        return self.active_session


def test_graph_only_results_can_enter_top_k(paper_factory):
    neural = [paper_factory(f"n{i}", 1.0 - i / 100) for i in range(1, 11)]
    symbolic = [paper_factory(f"s{i}", 1.0 - i / 100) for i in range(1, 11)]

    results = fuse_results(neural, symbolic, top_k=10)

    assert len(results) == 10
    assert any(item["source"] == "symbolic" for item in results)
    assert any(item["source"] == "neural" for item in results)


def test_symbolic_expansion_excludes_seed_self_matches_and_counts_distinct_seeds():
    driver = EmptyGraphDriver()

    assert symbolic_expand(driver, ["seed-1", "seed-2"]) == []
    assert "WHERE related.id <> pid" in driver.active_session.query
    assert "count(DISTINCT pid) AS connections" in driver.active_session.query


def test_overlap_is_labelled_both_and_boosted(paper_factory):
    neural = [paper_factory("shared"), paper_factory("neural-only")]
    symbolic = [paper_factory("shared"), paper_factory("graph-only")]

    results = fuse_results(neural, symbolic, top_k=3)

    assert results[0]["id"] == "shared"
    assert results[0]["source"] == "both"
    assert "neural_rank" in results[0]
    assert "graph_rank" in results[0]
    assert results[0]["score"] <= 1.0


def test_fusion_does_not_mutate_input_results(paper_factory):
    neural = [paper_factory("n1")]
    symbolic = [paper_factory("s1")]

    fuse_results(neural, symbolic, top_k=2)

    assert "neural_rank" not in neural[0]
    assert "graph_rank" not in symbolic[0]


def test_diagnostics_explain_kept_and_dropped_candidates(paper_factory):
    neural = [paper_factory("shared"), paper_factory("neural-only")]
    symbolic = [
        {**paper_factory("shared"), "graph_connections": 3},
        {**paper_factory("graph-only"), "graph_connections": 2},
        {**paper_factory("dropped"), "graph_connections": 1},
    ]
    final = fuse_results(neural, symbolic, top_k=2)

    report = build_retrieval_diagnostics(
        neural,
        symbolic,
        final,
        {"shared": 4, "neural-only": 0, "graph-only": 2, "dropped": 1},
    )
    rows = {row["id"]: row for row in report["candidates"]}

    assert report["source_distribution"]["both"] == 1
    assert rows["shared"]["citation_degree"] == 4
    assert rows["shared"]["decision"] == "kept in final top-k"
    assert rows["dropped"]["decision"] == "dropped below final cutoff"
