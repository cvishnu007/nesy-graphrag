"""Helpers behind reproduce.py: Holm adjustment, Fisher test, rebuilt arm keys, reranker order."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_holm_step_down_is_monotone_and_capped():
    from src.evaluation.holm_recheck import holm

    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert adjusted == {"a": 0.04, "c": 0.09, "b": 0.09, "d": 0.5}  # 4*0.01, 3*0.03, max(2*0.04, 0.09), 1*0.5
    assert holm({"x": 0.6, "y": 0.7}) == {"x": 1.0, "y": 1.0}


def test_fisher_two_sided_matches_known_values():
    from src.evaluation.reproduce import fisher_two_sided

    assert round(fisher_two_sided(5, 9, 5, 56), 4) == 0.0029  # scipy.stats.fisher_exact gives 0.00287
    assert round(fisher_two_sided(3, 3, 0, 3), 4) == 0.1  # classic tea-tasting table
    assert fisher_two_sided(1, 2, 1, 2) == pytest.approx(1.0)  # summed probabilities may be 0.9999999999999999


def test_rebuilt_arm_keys_reproduce_saved_reports(monkeypatch):
    monkeypatch.chdir(ROOT)
    spec = importlib.util.spec_from_file_location("rebuild", ROOT / "evaluation/human_review/rebuild_arm_keys.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from src.evaluation import human_ratings as hr

    keys = module.rebuilt_keys()
    review = ROOT / "evaluation/human_review"
    sheets = {n: hr.read(review / f"{n}.csv", hr.HYPOTHESIS_FIELDS, {"1", "2", "3", "4", "5"})
              for n in ("hypothesis_sheet_rated_cv", "hypothesis_sheet_rated_rater3")}
    saved = json.loads((review / "hypothesis_report_cv_rater3.json").read_text())
    report = hr.hypothesis_report(sheets, keys["hypothesis"])
    assert report["arms"] == saved["arms"] and report["graph_minus_vector"] == saved["graph_minus_vector"]
    round2 = {"contradiction_round2_sheet_labeled_cv": hr.read(
        review / "contradiction_round2_sheet_labeled_cv.csv", ("label_0_1_2",), {"0", "1", "2"})}
    saved = json.loads((review / "contradiction_round2_report_cv.json").read_text())
    assert hr.contradiction_report(round2, keys["contradiction_round2"])["arms"] == saved["arms"]


def test_rerank_order_sorts_by_score_and_keeps_ties_stable():
    from src.evaluation.reranker_baseline import rerank_order, variant_name

    assert rerank_order(["a", "b", "c", "d"], [0.1, 0.9, 0.1, 0.5]) == ["b", "d", "a", "c"]
    assert variant_name("cross-encoder/ms-marco-MiniLM-L-6-v2", "bm25") == "rerank_ms_marco_minilm_l_6_v2__bm25"


def test_regrade_round_robin_covers_every_topic_first():
    from src.evaluation.regrade_stratified import complete_queries, round_robin

    ids = [f"V2-{t:02d}-{n}" for t in range(1, 12) for n in range(1, 5)]
    order = round_robin(ids)
    assert order[:11] == [f"V2-{t:02d}-1" for t in range(1, 12)] and order[11] == "V2-01-2"
    rows = [{"query_id": "q1", "item_id": "q1/a"}, {"query_id": "q1", "item_id": "q1/b"},
            {"query_id": "q2", "item_id": "q2/a"}]
    assert complete_queries(rows, {"q1/a": 2, "q2/a": 0}) == ["q2"]
