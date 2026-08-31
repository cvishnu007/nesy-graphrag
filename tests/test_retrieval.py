from src.pipeline.retrieval import (
    build_retrieval_diagnostics,
    filter_symbolic_candidates,
    fuse_results,
    query_term_coverage,
    relevant_symbolic_expand,
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


def test_query_term_coverage_ignores_connector_words(paper_factory):
    paper = paper_factory("p1")
    paper["title"] = "Graph neural networks for classification"
    paper["abstract"] = "A node-level learning method."

    assert query_term_coverage(
        "graph neural networks for node classification",
        paper,
    ) == 1.0


def test_filter_keeps_high_semantic_graph_candidate(paper_factory):
    candidate = {
        **paper_factory("high-semantic"),
        "graph_connections": 1,
    }

    result = filter_symbolic_candidates(
        "graph neural networks",
        [candidate],
        {"high-semantic": 0.86},
    )

    assert [paper["id"] for paper in result] == ["high-semantic"]
    assert result[0]["graph_filter_reason"] == "high_semantic_similarity"


def test_filter_keeps_well_supported_multi_seed_candidate(paper_factory):
    candidate = {
        **paper_factory("supported"),
        "title": "Graph neural networks",
        "abstract": "Neural graph learning.",
        "graph_connections": 10,
    }

    result = filter_symbolic_candidates(
        "graph neural networks",
        [candidate],
        {"supported": 0.80},
    )

    assert [paper["id"] for paper in result] == ["supported"]
    assert result[0]["graph_filter_reason"] == "strong_multi_seed_support"


def test_filter_rejects_weak_or_missing_semantic_match(paper_factory):
    weak = {
        **paper_factory("weak"),
        "title": "Graph neural networks",
        "graph_connections": 2,
    }
    missing = {
        **paper_factory("missing"),
        "title": "Graph neural networks",
        "graph_connections": 20,
    }

    assert filter_symbolic_candidates(
        "graph neural networks",
        [weak, missing],
        {"weak": 0.74},
    ) == []


def test_filter_does_not_mutate_symbolic_candidates(paper_factory):
    candidate = {
        **paper_factory("p1"),
        "graph_connections": 1,
    }

    filter_symbolic_candidates("graph learning", [candidate], {"p1": 0.90})

    assert "semantic_similarity" not in candidate


def test_relevant_symbolic_expand_scores_and_filters_graph_results(
    monkeypatch,
    paper_factory,
):
    from src.pipeline import retrieval

    candidates = [
        {**paper_factory("keep"), "graph_connections": 1},
        {**paper_factory("drop"), "graph_connections": 1},
    ]
    monkeypatch.setattr(retrieval, "symbolic_expand", lambda driver, ids: candidates)
    monkeypatch.setattr(
        retrieval,
        "score_papers_against_query",
        lambda query, ids: {"keep": 0.90, "drop": 0.20},
    )

    result = relevant_symbolic_expand(object(), "graph learning", ["seed"])

    assert [paper["id"] for paper in result] == ["keep"]
