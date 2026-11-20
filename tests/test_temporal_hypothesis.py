import numpy as np

from src.evaluation.temporal_hypothesis import rank_scores, rrf_scores, task_metrics


def test_rank_scores_average_ties():
    assert list(rank_scores(np.array([3.0, 1.0, 3.0]))) == [1.5, 3.0, 1.5]


def test_task_metrics_perfect_and_worst():
    best = task_metrics(np.array([3.0, 2.0, 1.0, 0.0]), np.array([True, True, False, False]))
    worst = task_metrics(np.array([0.0, 1.0, 2.0, 3.0]), np.array([True, True, False, False]))
    assert best["ap"] == 1.0 and best["auroc"] == 1.0 and worst["auroc"] == 0.0


def test_rrf_prefers_items_ranked_high_in_both():
    fused = rrf_scores([np.array([3.0, 2.0, 1.0]), np.array([1.0, 3.0, 2.0])])
    assert int(np.argmax(fused)) == 1


def test_propose_skips_known_and_duplicates():
    from src.evaluation.temporal_generator import propose

    assert propose(["a", "b", "a", "c", "d"], known={"b"}, limit=2) == ["a", "c"]


def test_entity_types_can_ignore_post_cutoff_mentions():
    from src.evaluation.temporal_hypothesis import entity_types

    mentions = [("old1", "gnn", "Method"), ("new1", "gnn", "Task"), ("new2", "gnn", "Task")]
    assert entity_types(mentions) == {"gnn": "Task"}  # majority over every mention (frozen default)
    assert entity_types(mentions, papers={"old1"}) == {"gnn": "Method"}  # pre-cutoff mentions only
