from src.pipeline.hypothesis_bridge import bridge_score, is_specific


def test_rarer_shared_problems_and_novel_methods_score_higher():
    rare = bridge_score([3], [5], 0.8, 47619)
    common = bridge_score([400], [5], 0.8, 47619)
    no_method = bridge_score([3], [], 0.8, 47619)
    assert rare > common and rare > no_method > 0


def test_specificity_filter_drops_lone_common_words():
    assert not is_specific("analysis") and not is_specific("pytorch")
    assert is_specific("graph contrastive learning") and is_specific("nsl-kdd") and is_specific("mpnn")
