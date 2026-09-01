import pytest

from src.evaluation.hypothesis_metrics import (
    evaluate_hypotheses,
    observed_agreement,
    rating_accepted,
    rating_summary,
    reviewer_agreement,
    weighted_cohens_kappa,
)


def rating(reviewer="R1", **overrides):
    value = {
        "reviewer_id": reviewer, "evidence": 3, "novelty": 5,
        "feasibility": 3, "specificity": 3, "usefulness": 5,
    }
    value.update(overrides)
    return value


def test_pdf_acceptance_rule():
    assert rating_accepted(rating()) is True
    assert rating_accepted(rating(novelty=1)) is False
    assert rating_accepted(rating(evidence=1)) is False


def test_dimension_and_aggregate_statistics():
    result = rating_summary([rating(), rating("R2", novelty=3, usefulness=3)])
    assert result["dimensions"]["novelty"]["mean"] == 4.0
    assert result["dimensions"]["novelty"]["std"] == 1.0
    assert result["rating_count"] == 2
    assert result["acceptance_rate"] == 1.0


def test_empty_ratings_return_no_fabricated_statistics():
    result = rating_summary([])
    assert result["aggregate_score_mean"] is None
    assert result["acceptance_rate"] is None


def test_observed_agreement_and_weighted_kappa():
    assert observed_agreement([1, 3, 5], [1, 3, 5]) == 1.0
    assert weighted_cohens_kappa([1, 3, 5], [1, 3, 5]) == 1.0
    assert weighted_cohens_kappa([1], [1]) is None


def test_agreement_requires_shared_ratings():
    result = reviewer_agreement([{"ratings": [rating("R1")]}])
    assert result["status"] == "insufficient_data"


def test_multiple_reviewers_and_hns_relationship():
    items = [
        {"hns": 0.2, "model_feasibility": "MEDIUM", "ratings": [rating("R1", novelty=1), rating("R2", novelty=1)]},
        {"hns": 0.8, "model_feasibility": "MEDIUM", "ratings": [rating("R1", novelty=5), rating("R2", novelty=5)]},
    ]
    result = evaluate_hypotheses(items)
    assert result["annotation_pass_agreement"]["status"] == "available"
    assert result["reference_model_feasibility_agreement"]["status"] == "available"
    assert result["reference_model_feasibility_agreement"]["observed_agreement"] == 1.0
    assert result["hns_reference_novelty"]["pearson_correlation"] == pytest.approx(1.0)
    assert result["reference_annotation_source"] == "AI-generated"
    assert result["human_ground_truth"] is False


@pytest.mark.parametrize("score", [0, 2, 4, 6, True])
def test_invalid_rating_score(score):
    with pytest.raises(ValueError):
        rating_summary([rating(novelty=score)])
