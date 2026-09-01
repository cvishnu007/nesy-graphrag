import numpy as np

from src.evaluation.embedding_comparison import compare_embedding_models


class FakeEncoder:
    def __init__(self, reverse=False):
        self.reverse = reverse

    def encode(self, texts, **kwargs):
        values = []
        for text in texts:
            number = float(sum(ord(char) for char in text) % 17 + 1)
            vector = [number, 1.0]
            values.append(list(reversed(vector)) if self.reverse else vector)
        return np.asarray(values, dtype=float)


def benchmark():
    return {
        "benchmark_version": "retrieval-ai-reference-v1",
        "queries": [
            {
                "query_id": "Q1", "split": "dev", "query": "graphs",
                "judgments": {"P1": 2, "P2": 0},
            }
        ],
    }


def corpus():
    return [
        {"id": "P1", "title": "Graph paper", "abstract": "graphs"},
        {"id": "P2", "title": "Other paper", "abstract": "other"},
    ]


def test_embedding_models_receive_identical_queries_and_candidates():
    result = compare_embedding_models(
        benchmark(), corpus(),
        {"specter": FakeEncoder(), "minilm": FakeEncoder(reverse=True)},
        split="dev", top_k=20,
    )

    assert result["models"]["specter"]["query_ids"] == ["Q1"]
    assert result["models"]["specter"]["query_ids"] == result["models"]["minilm"]["query_ids"]
    assert result["models"]["specter"]["candidate_ids"] == result["models"]["minilm"]["candidate_ids"]
    assert result["controlled_variables"]["only_changed_component"] == "embedding_model"


def test_embedding_comparison_reports_standard_ir_metrics():
    result = compare_embedding_models(
        benchmark(), corpus(), {"specter": FakeEncoder()},
        split="dev", top_k=20,
    )

    metrics = result["models"]["specter"]["summary"]["metrics"]
    assert {"mrr", "map", "precision@5", "recall@10", "ndcg@20"}.issubset(metrics)
