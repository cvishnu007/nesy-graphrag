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


def test_neighbour_list_drops_citing_papers_and_optional_near_duplicates():
    import numpy as np

    from src.evaluation.citation_recommendation import neighbour_list

    ids = np.array(["dup", "citer", "a", "b"])
    sims = np.array([0.999, 0.95, 0.9, 0.8])
    order = [0, 1, 2, 3]
    assert [p for p, _ in neighbour_list(order, sims, ids, {"citer"})] == ["dup", "a", "b"]
    kept = neighbour_list(order, sims, ids, {"citer"}, near_duplicate_cosine=0.995)
    assert kept == [("a", 0.9), ("b", 0.8)]


def test_per_query_table_keeps_reported_metrics_only():
    from src.evaluation.citation_recommendation import PER_QUERY_METRICS, per_query_table

    row = {"query_id": "q1", "ndcg@10": 0.5, "recall@20": 0.25, "precision@10": 0.1, "map": 0.2, "mrr": 1.0,
           "ndcg@5": 0.4}
    table = per_query_table({"bge": [row]})
    assert table == {"bge": {"q1": {m: row[m] for m in PER_QUERY_METRICS}}}
