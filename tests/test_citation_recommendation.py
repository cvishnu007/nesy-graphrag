from src.evaluation.citation_recommendation import cocitation_ranking, rrf, select_queries


def test_cocitation_scores_papers_cited_by_similar_neighbours():
    cites = {"n1": {"a", "b"}, "n2": {"a"}, "n3": {"c"}}
    ranking = cocitation_ranking(["n1", "n2", "n3"], [0.9, 0.8, 0.1], cites, allowed=lambda c: c != "b")
    assert ranking == ["a", "c"]  # a is co-cited by two close neighbours; b is filtered out


def test_rrf_weights_lists():
    assert rrf([["x", "y"], ["y", "x"]], [1.0, 2.0])[0] == "y"


def test_query_selection_requires_older_cited_papers():
    corpus = [{"id": "q", "year": 2023}] + [{"id": f"c{i}", "year": 2020} for i in range(6)] + [{"id": "f", "year": 2025}]
    cites = {"q": {f"c{i}" for i in range(4)} | {"f"}}
    dev, test, eligible = select_queries(corpus, cites, count=3)
    assert eligible == 0  # only 4 older targets; the 2025 paper does not count
    cites["q"].add("c5")
    assert select_queries(corpus, cites, count=3)[2] == 1
