"""The blinded sheets' item mappings must be rebuildable from committed files alone."""
import importlib.util
from collections import Counter


def _load(name):
    spec = importlib.util.spec_from_file_location(name, f"evaluation/human_review/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relevance_subset_mapping_is_deterministic():
    builder = _load("build_relevance_subset")
    mapping = builder.subset_mapping()
    assert len(mapping) == 50 and len(set(mapping.values())) == 50
    assert mapping == builder.subset_mapping()


def test_judge_transfer_sheet_covers_ten_topics_evenly():
    builder = _load("build_judge_transfer_sheet")
    mapping = builder.subset_mapping()
    assert len(mapping) == 100 and len(set(mapping.values())) == 100
    topics = builder.queries()
    counts = Counter(topics[item.split("/")[0]]["topic"] for item in mapping.values())
    assert len(counts) == 10 and set(counts.values()) == {10}
    assert builder.EXCLUDED_TOPIC not in counts
    labels = builder.judge_labels()
    assert Counter(labels[i] for i in mapping.values()) == {0: 40, 1: 20, 2: 40}
