"""Standard information-retrieval metrics for the judged benchmark."""

from math import log2
from statistics import fmean, pstdev
from typing import Iterable, Mapping


def _validate_k(k: int) -> None:
    """Require a positive integer cutoff."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be greater than zero")


def _unique_ranked_ids(ranked_ids: Iterable[str]) -> list[str]:
    """Remove duplicate paper IDs while keeping their first position."""
    unique = []
    seen = set()

    for paper_id in ranked_ids:
        if paper_id not in seen:
            seen.add(paper_id)
            unique.append(paper_id)

    return unique


def _relevant_ids(judgments: Mapping[str, int]) -> set[str]:
    """Grades 1 and 2 are relevant for binary metrics."""
    return {
        paper_id
        for paper_id, grade in judgments.items()
        if grade > 0
    }


def precision_at_k(
    ranked_ids: Iterable[str],
    judgments: Mapping[str, int],
    k: int,
) -> float:
    """Relevant papers retrieved in the top K, divided by K."""
    _validate_k(k)
    top_k = _unique_ranked_ids(ranked_ids)[:k]
    relevant = _relevant_ids(judgments)
    hits = sum(paper_id in relevant for paper_id in top_k)
    return hits / k


def recall_at_k(
    ranked_ids: Iterable[str],
    judgments: Mapping[str, int],
    k: int,
) -> float:
    """Relevant papers retrieved in the top K, divided by all known relevant papers."""
    _validate_k(k)
    relevant = _relevant_ids(judgments)

    if not relevant:
        return 0.0

    top_k = _unique_ranked_ids(ranked_ids)[:k]
    hits = sum(paper_id in relevant for paper_id in top_k)
    return hits / len(relevant)


def hit_rate_at_k(
    ranked_ids: Iterable[str],
    judgments: Mapping[str, int],
    k: int,
) -> float:
    """Return 1 when the top K contains a relevant paper, otherwise 0."""
    _validate_k(k)
    relevant = _relevant_ids(judgments)
    top_k = _unique_ranked_ids(ranked_ids)[:k]
    return float(any(paper_id in relevant for paper_id in top_k))


def reciprocal_rank(
    ranked_ids: Iterable[str],
    judgments: Mapping[str, int],
) -> float:
    """Return the reciprocal rank of the first relevant paper."""
    relevant = _relevant_ids(judgments)

    for rank, paper_id in enumerate(_unique_ranked_ids(ranked_ids), start=1):
        if paper_id in relevant:
            return 1.0 / rank

    return 0.0


def average_precision(
    ranked_ids: Iterable[str],
    judgments: Mapping[str, int],
) -> float:
    """Average precision at every relevant hit."""
    relevant = _relevant_ids(judgments)

    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, paper_id in enumerate(_unique_ranked_ids(ranked_ids), start=1):
        if paper_id in relevant:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant)


def ndcg_at_k(
    ranked_ids: Iterable[str],
    judgments: Mapping[str, int],
    k: int,
) -> float:
    """Calculate graded NDCG using gain 2^grade - 1."""
    _validate_k(k)
    top_k = _unique_ranked_ids(ranked_ids)[:k]

    dcg = sum(
        ((2 ** judgments.get(paper_id, 0)) - 1) / log2(rank + 1)
        for rank, paper_id in enumerate(top_k, start=1)
    )

    ideal_grades = sorted(judgments.values(), reverse=True)[:k]
    ideal_dcg = sum(
        ((2 ** grade) - 1) / log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def unjudged_rate_at_k(
    ranked_ids: Iterable[str],
    judgments: Mapping[str, int],
    k: int,
) -> float:
    """Fraction of returned top-K papers without a judgment."""
    _validate_k(k)
    top_k = _unique_ranked_ids(ranked_ids)[:k]

    if not top_k:
        return 0.0

    unjudged = sum(paper_id not in judgments for paper_id in top_k)
    return unjudged / len(top_k)


def evaluate_ranking(
    ranked_ids: Iterable[str],
    judgments: Mapping[str, int],
    k_values: Iterable[int] = (5, 10, 20),
) -> dict[str, float]:
    """Calculate all metrics for one query ranking."""
    ranked_ids = _unique_ranked_ids(ranked_ids)
    k_values = tuple(k_values)

    results = {
        "mrr": reciprocal_rank(ranked_ids, judgments),
        "map": average_precision(ranked_ids, judgments),
    }

    for k in k_values:
        _validate_k(k)
        results[f"precision@{k}"] = precision_at_k(
            ranked_ids, judgments, k
        )
        results[f"recall@{k}"] = recall_at_k(
            ranked_ids, judgments, k
        )
        results[f"hit_rate@{k}"] = hit_rate_at_k(
            ranked_ids, judgments, k
        )
        results[f"ndcg@{k}"] = ndcg_at_k(
            ranked_ids, judgments, k
        )
        results[f"unjudged_rate@{k}"] = unjudged_rate_at_k(
            ranked_ids, judgments, k
        )

    return results


def aggregate_query_metrics(
    rows: Iterable[Mapping[str, float]],
) -> dict:
    """Macro-average numeric metrics across queries."""
    rows = list(rows)

    if not rows:
        return {
            "query_count": 0,
            "metrics": {},
        }

    metric_names = sorted(
        key
        for key in rows[0]
        if all(
            isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
            for row in rows
        )
    )

    metrics = {}

    for name in metric_names:
        values = [float(row[name]) for row in rows]
        metrics[name] = {
            "mean": fmean(values),
            "std": pstdev(values),
        }

    return {
        "query_count": len(rows),
        "metrics": metrics,
    }