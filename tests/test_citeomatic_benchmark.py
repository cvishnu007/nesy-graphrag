from src.evaluation.citeomatic_benchmark import gold, metrics, split_ids, summarise


def test_metrics_follow_citeomatic_definitions():
    m = metrics(["a", "x", "b"], {"a", "b", "c"})
    assert m == {"p@20": 2 / 20, "r@20": 2 / 3, "rr": 1.0}
    assert metrics(["x", "y", "b"], {"b"})["rr"] == 1 / 3
    assert metrics(["x"], {"b"})["rr"] == 0.0


def test_summary_f1_is_harmonic_mean_of_corpus_level_p_and_r():
    s = summarise({"q1": {"p@20": 0.2, "r@20": 0.4, "rr": 1.0}, "q2": {"p@20": 0.0, "r@20": 0.0, "rr": 0.5}})
    p, r = 0.1, 0.2
    assert s["f1@20"] == round(2 * p * r / (p + r), 4) and s["mrr"] == 0.75


def test_gold_keeps_older_training_citations_only():
    papers = {"q": {"year": 2010, "out": {"a", "b", "c", "q"}}, "a": {"year": 2005, "out": set()},
              "b": {"year": 2012, "out": set()}, "c": {"year": 2009, "out": set()}}
    assert gold(papers, "q", train={"a", "b"}) == {"a"}
    assert split_ids({"x": {"year": 2009}, "y": {"year": 2006}}, "dblp", "test") == ["x"]
