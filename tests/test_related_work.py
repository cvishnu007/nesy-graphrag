from src.pipeline.related_work import fuse


def test_fuse_combines_sources_and_excludes():
    out = fuse(["a", "b"], ["b", "c"], ["c", "b"], top_k=3, exclude={"a"})
    ids = [pid for pid, _, _ in out]
    assert ids[0] == "b" and "a" not in ids
    assert out[0][2] == {"bge": 2, "bm25": 1, "cocitation": 2}


def test_graph_weight_zero_ignores_cocitation_order():
    out = fuse(["x"], ["y"], ["z"], top_k=3, graph_weight=0.0)
    assert [pid for pid, _, _ in out][-1] == "z"
