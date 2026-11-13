"""Cross-check ir_metrics against the reference trec_eval implementation."""
import math
import random

import pytest

pytrec_eval = pytest.importorskip("pytrec_eval")

from src.evaluation.ir_metrics import evaluate_ranking  # noqa: E402


def test_metrics_match_trec_eval_on_random_rankings():
    rng = random.Random(7)
    for trial in range(50):
        docs = [f"d{i}" for i in range(40)]
        qrels = {d: rng.choice([0, 0, 1, 2]) for d in rng.sample(docs, 25)}
        if not any(qrels.values()):
            qrels[docs[0]] = 2
        ranked = rng.sample(docs, 20)
        reference = pytrec_eval.RelevanceEvaluator(
            {"q": qrels}, {"P_10", "recall_20", "recip_rank", "ndcg_cut_10"}
        ).evaluate({"q": {d: float(20 - i) for i, d in enumerate(ranked)}})["q"]
        ours = evaluate_ranking(ranked, qrels, (10, 20))
        assert ours["precision@10"] == pytest.approx(reference["P_10"])
        assert ours["recall@20"] == pytest.approx(reference["recall_20"])
        assert ours["mrr"] == pytest.approx(reference["recip_rank"])
        # trec_eval uses linear gain; ir_metrics reports exponential gain (2^g - 1).
        dcg = sum(qrels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
        ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(qrels.values(), reverse=True)[:10]))
        assert (dcg / ideal if ideal else 0.0) == pytest.approx(reference["ndcg_cut_10"])
