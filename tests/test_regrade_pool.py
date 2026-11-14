import json

from src.evaluation.regrade_pool import compare, pool_rows, score


def test_pool_rows_union_frozen_and_extension_with_hash_order(tmp_path):
    benchmark = {"queries": [{"query_id": "Q1", "split": "dev", "query": "gnn", "judgments": {"a": 2, "b": 0}}]}
    candidates = tmp_path / "c.csv"
    candidates.write_text("query_id,split,query,paper_id,title,abstract\nQ1,dev,gnn,a,A,aa\nQ1,dev,gnn,b,B,bb\n"
                          "Q1,dev,gnn,c,C,cc\n", encoding="utf-8")
    ext_dir = tmp_path / "variant_ablation_x"
    ext_dir.mkdir()
    (ext_dir / "extension_dev_checkpoint.json").write_text(json.dumps({"judgments": {"Q1/c": {"grade": 1}}}))
    rows = pool_rows(benchmark, str(candidates), str(tmp_path / "variant_ablation*" / "extension_*_checkpoint.json"))
    assert sorted(r["paper_id"] for r in rows) == ["a", "b", "c"]
    assert all(r["query"] == "gnn" and r["paper_title"] for r in rows)
    assert pool_rows(benchmark, str(candidates), str(tmp_path / "none*")) != []


def test_score_and_compare_report_paired_deltas():
    labels = {"Q1": {"a": 2, "b": 0}, "Q2": {"a": 0, "b": 2}}
    rankings = {"bm25": {"Q1": ["a", "b"], "Q2": ["b", "a"]},
                "production_absolute": {"Q1": ["b", "a"], "Q2": ["a", "b"]}}
    summary = score(rankings, labels, ["Q1", "Q2"])
    assert summary["bm25"]["ndcg@10"] > summary["production_absolute"]["ndcg@10"]
    result = compare(summary)["bm25_vs_production_absolute"]
    assert result["wins"] == 2 and result["pair_count"] == 2


def test_stratified_pool_is_union_of_top20_in_hash_order():
    from src.evaluation.stratified_benchmark import pool

    queries = [{"query_id": "V1", "query": "q"}]
    rankings = {"a": {"V1": ["p1", "p2"]}, "b": {"V1": ["p2", "p3"] + [f"x{i}" for i in range(30)]}}
    corpus = {pid: {"clean_title": pid, "clean_abstract": ""} for pid in ["p1", "p2", "p3"] + [f"x{i}" for i in range(30)]}
    rows = pool(queries, rankings, corpus)
    assert len(rows) == 3 + 18  # b contributes its top 20 only
    assert len({r["paper_id"] for r in rows}) == len(rows)


def test_human_ratings_reports_arm_means_and_precision():
    from src.evaluation.human_ratings import contradiction_report, hypothesis_report

    key = {"items": [{"item_id": "H1", "query_id": "Q", "arm": "graph_structural_hole"},
                     {"item_id": "H2", "query_id": "Q", "arm": "vector_neighbour"}]}
    sheet = {"H1": {"relevance_1_5": 4, "novelty_1_5": 5, "promise_1_5": 4},
             "H2": {"relevance_1_5": 5, "novelty_1_5": 2, "promise_1_5": 3}}
    report = hypothesis_report({"a": sheet}, key)
    assert report["arms"]["graph_structural_hole"]["novelty_1_5"] == 5
    assert report["graph_minus_vector"]["novelty_1_5"]["mean_delta"] == 3
    ckey = {"items": [{"item_id": "C1", "arm": "top_ranked"}, {"item_id": "C2", "arm": "top_ranked"},
                      {"item_id": "C3", "arm": "lower_ranked_control"}]}
    labels = {"C1": {"label_0_1_2": 2}, "C2": {"label_0_1_2": 0}, "C3": {"label_0_1_2": 1}}
    result = contradiction_report({"a": labels}, ckey)["arms"]
    assert result["top_ranked"]["precision_contradiction"] == 0.5
    assert result["lower_ranked_control"]["precision_contradiction_or_tension"] == 1.0


def test_round2_hypothesis_item_counts_for_every_arm_that_chose_it():
    from src.evaluation.human_ratings import hypothesis_round2_report

    key = {"items": [{"item_id": "A", "query_id": "Q", "arms": ["bridge_m10", "bridge_m10_specific"]},
                     {"item_id": "B", "query_id": "Q", "arms": ["vector_neighbour"]}]}
    sheet = {"A": {"relevance_1_5": 4, "novelty_1_5": 4, "promise_1_5": 4},
             "B": {"relevance_1_5": 5, "novelty_1_5": 1, "promise_1_5": 2}}
    report = hypothesis_round2_report({"x": sheet}, key)
    assert report["arms"]["bridge_m10_specific"]["novelty_1_5"] == 4
    assert report["comparisons"]["bridge_m10_minus_vector_neighbour"]["novelty_1_5"]["mean_delta"] == 3
