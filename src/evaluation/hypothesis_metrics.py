"""Hypothesis-rubric aggregation for declared reference annotations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from math import sqrt
from statistics import fmean, pstdev

from src.evaluation.reasoning_benchmark_io import (
    HYPOTHESIS_DIMENSIONS,
    HYPOTHESIS_SCORES,
)


SCORE_ORDER = (1, 3, 5)


def validate_rating(rating: Mapping) -> None:
    if not isinstance(rating, Mapping):
        raise TypeError("rating must be an object")
    for dimension in HYPOTHESIS_DIMENSIONS:
        score = rating.get(dimension)
        if isinstance(score, bool) or score not in HYPOTHESIS_SCORES:
            raise ValueError(f"{dimension} must be one of {sorted(HYPOTHESIS_SCORES)}")


def rating_accepted(rating: Mapping) -> bool:
    """Apply the PDF's predeclared example acceptance rule."""
    validate_rating(rating)
    return (
        rating["evidence"] >= 3
        and rating["feasibility"] >= 3
        and rating["specificity"] >= 3
        and all(rating[name] != 1 for name in HYPOTHESIS_DIMENSIONS)
    )


def rating_summary(ratings: Iterable[Mapping]) -> dict:
    rows = list(ratings)
    for row in rows:
        validate_rating(row)
    dimensions = {}
    for name in HYPOTHESIS_DIMENSIONS:
        values = [row[name] for row in rows]
        dimensions[name] = {
            "count": len(values),
            "mean": fmean(values) if values else None,
            "std": pstdev(values) if values else None,
        }
    aggregate_values = [fmean(row[name] for name in HYPOTHESIS_DIMENSIONS) for row in rows]
    return {
        "rating_count": len(rows),
        "dimensions": dimensions,
        "aggregate_score_mean": fmean(aggregate_values) if aggregate_values else None,
        "aggregate_score_std": pstdev(aggregate_values) if aggregate_values else None,
        "acceptance_rate": (
            sum(rating_accepted(row) for row in rows) / len(rows) if rows else None
        ),
    }


def observed_agreement(first: Sequence[int], second: Sequence[int]) -> float | None:
    if len(first) != len(second):
        raise ValueError("rating sequences must have equal length")
    if not first:
        return None
    if any(value not in HYPOTHESIS_SCORES for value in [*first, *second]):
        raise ValueError("ratings must use scores 1, 3, or 5")
    return sum(left == right for left, right in zip(first, second)) / len(first)


def weighted_cohens_kappa(first: Sequence[int], second: Sequence[int]) -> float | None:
    """Quadratic weighted Cohen's kappa for the ordered 1/3/5 scale."""
    if len(first) != len(second):
        raise ValueError("rating sequences must have equal length")
    if len(first) < 2:
        return None
    if any(value not in HYPOTHESIS_SCORES for value in [*first, *second]):
        raise ValueError("ratings must use scores 1, 3, or 5")
    positions = {score: index for index, score in enumerate(SCORE_ORDER)}
    size = len(SCORE_ORDER)
    observed = [[0.0] * size for _ in range(size)]
    for left, right in zip(first, second):
        observed[positions[left]][positions[right]] += 1
    total = len(first)
    row_totals = [sum(row) for row in observed]
    column_totals = [sum(observed[row][column] for row in range(size)) for column in range(size)]
    weighted_observed = weighted_expected = 0.0
    denominator = (size - 1) ** 2
    for row in range(size):
        for column in range(size):
            weight = ((row - column) ** 2) / denominator
            weighted_observed += weight * observed[row][column] / total
            weighted_expected += weight * (row_totals[row] * column_totals[column]) / (total * total)
    if weighted_expected == 0:
        return 1.0 if weighted_observed == 0 else None
    return 1 - weighted_observed / weighted_expected


def reviewer_agreement(hypotheses: Iterable[Mapping]) -> dict:
    """Report pairwise agreement only for reviewers with shared hypotheses."""
    pair_values = defaultdict(lambda: defaultdict(lambda: ([], [])))
    for item in hypotheses:
        ratings = item.get("ratings", [])
        for rating in ratings:
            validate_rating(rating)
        ordered = sorted(ratings, key=lambda row: row.get("reviewer_id", ""))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1:]:
                pair = (left.get("reviewer_id"), right.get("reviewer_id"))
                if not all(isinstance(value, str) and value for value in pair):
                    raise ValueError("reviewer_id must be a non-empty string")
                for dimension in HYPOTHESIS_DIMENSIONS:
                    pair_values[pair][dimension][0].append(left[dimension])
                    pair_values[pair][dimension][1].append(right[dimension])
    pairs = []
    for reviewers, dimensions in sorted(pair_values.items()):
        dimension_results = {}
        for dimension, (first, second) in dimensions.items():
            dimension_results[dimension] = {
                "shared_count": len(first),
                "observed_agreement": observed_agreement(first, second),
                "weighted_kappa": weighted_cohens_kappa(first, second),
            }
        pairs.append({"reviewers": list(reviewers), "dimensions": dimension_results})
    sufficient = bool(pairs) and any(
        result["shared_count"] >= 2
        for pair in pairs for result in pair["dimensions"].values()
    )
    return {
        "status": "available" if sufficient else "insufficient_data",
        "reviewer_pairs": pairs,
    }


def pearson_correlation(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second):
        raise ValueError("value sequences must have equal length")
    if len(first) < 2:
        return None
    mean_first, mean_second = fmean(first), fmean(second)
    numerator = sum((x - mean_first) * (y - mean_second) for x, y in zip(first, second))
    first_sum = sum((x - mean_first) ** 2 for x in first)
    second_sum = sum((y - mean_second) ** 2 for y in second)
    denominator = sqrt(first_sum * second_sum)
    return numerator / denominator if denominator else None


def evaluate_hypotheses(hypotheses: Iterable[Mapping]) -> dict:
    items = list(hypotheses)
    ratings = [rating for item in items for rating in item.get("ratings", [])]
    hns_values = []
    novelty_values = []
    for item in items:
        hns = item.get("hns")
        item_ratings = item.get("ratings", [])
        if isinstance(hns, (int, float)) and not isinstance(hns, bool) and item_ratings:
            hns_values.append(float(hns))
            novelty_values.append(fmean(rating["novelty"] for rating in item_ratings))
    model_scores = []
    reference_scores = []
    feasibility_map = {"LOW": 1, "MEDIUM": 3, "HIGH": 5}
    accepted_hypotheses = 0
    rated_hypotheses = 0
    for item in items:
        item_ratings = item.get("ratings", [])
        if item_ratings:
            rated_hypotheses += 1
            dimension_means = {
                name: fmean(rating[name] for rating in item_ratings)
                for name in HYPOTHESIS_DIMENSIONS
            }
            if (
                dimension_means["evidence"] >= 3
                and dimension_means["feasibility"] >= 3
                and dimension_means["specificity"] >= 3
                and all(value > 1 for value in dimension_means.values())
            ):
                accepted_hypotheses += 1
            model_value = feasibility_map.get(item.get("model_feasibility"))
            if model_value is not None:
                model_scores.append(model_value)
                reference_scores.append(min(SCORE_ORDER, key=lambda score: (abs(score - dimension_means["feasibility"]), score)))
    feasibility_agreement = {
        "pair_count": len(model_scores),
        "observed_agreement": observed_agreement(model_scores, reference_scores),
        "weighted_kappa": weighted_cohens_kappa(model_scores, reference_scores),
        "status": "available" if len(model_scores) >= 2 else "insufficient_data",
    }
    return {
        **rating_summary(ratings),
        "hypothesis_count": len(items),
        "rated_hypothesis_count": rated_hypotheses,
        "hypothesis_acceptance_rate": (
            accepted_hypotheses / rated_hypotheses if rated_hypotheses else None
        ),
        "reference_model_feasibility_agreement": feasibility_agreement,
        "annotation_pass_agreement": reviewer_agreement(items),
        "hns_reference_novelty": {
            "pair_count": len(hns_values),
            "pearson_correlation": pearson_correlation(hns_values, novelty_values),
            "status": "available" if len(hns_values) >= 2 else "insufficient_data",
        },
        "reference_annotation_source": "AI-generated",
        "human_ground_truth": False,
    }
