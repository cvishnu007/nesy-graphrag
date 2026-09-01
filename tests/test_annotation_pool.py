import pytest

from src.evaluation.annotation_pool import (
    AnnotationPoolError,
    finalize_annotation_pool,
    generate_annotation_pool,
    stable_example_id,
    validate_annotation_pool,
)


def contradiction_candidate(**overrides):
    value = {
        "split": "dev",
        "paper1": {"id": "paper-b", "title": "B", "abstract": "Abstract B"},
        "paper2": {"id": "paper-a", "title": "A", "abstract": "Abstract A"},
        "verdict": "CONTRADICTION", "confidence": 0.91, "candidate_score": 0.8,
    }
    value.update(overrides)
    return value


def test_contradiction_pool_is_canonical_stable_and_blinded():
    pool, sidecar = generate_annotation_pool("contradiction", [contradiction_candidate()])
    record = pool["records"][0]
    assert (record["paper1_id"], record["paper2_id"]) == ("paper-a", "paper-b")
    assert record["pair_id"] == stable_example_id("C", "paper-a", "paper-b")
    assert "verdict" not in record and "confidence" not in record
    assert sidecar["system_records"][0]["prediction"] == "CONTRADICTION"
    assert sidecar["system_records"][0]["confidence"] == 0.91


def test_generation_configuration_is_protected_system_only_metadata():
    configuration = {
        "primary": "alternate-model", "fallback": "alternate-model",
        "configuration_label": "revised", "revised_configuration": True,
    }
    pool, sidecar = generate_annotation_pool("contradiction", [
        contradiction_candidate(generation_configuration=configuration)
    ])
    assert "generation_configuration" not in pool["records"][0]
    assert sidecar["system_records"][0]["generation_configuration"] == configuration


def test_reversed_input_has_the_same_stable_id():
    first, _ = generate_annotation_pool("contradiction", [contradiction_candidate()])
    reversed_candidate = contradiction_candidate(
        paper1={"id": "paper-a", "title": "A", "abstract": "Abstract A"},
        paper2={"id": "paper-b", "title": "B", "abstract": "Abstract B"},
    )
    second, _ = generate_annotation_pool("contradiction", [reversed_candidate])
    assert first["records"][0]["pair_id"] == second["records"][0]["pair_id"]


def test_duplicate_unordered_pairs_are_rejected():
    reversed_candidate = contradiction_candidate(
        paper1={"id": "paper-a", "title": "A", "abstract": "Abstract A"},
        paper2={"id": "paper-b", "title": "B", "abstract": "Abstract B"},
    )
    with pytest.raises(AnnotationPoolError, match="Duplicate annotation example"):
        generate_annotation_pool("contradiction", [contradiction_candidate(), reversed_candidate])


def test_support_pool_separates_prediction_and_human_fields():
    candidate = {
        "split": "test", "query_id": "Q1", "claim": "A claim",
        "passage_id": "P1-S001", "passage_text": "Evidence", "paper_id": "P1",
        "support_label": "SUPPORTED", "confidence": 0.8, "model": "fixture-model",
    }
    pool, sidecar = generate_annotation_pool("support", [candidate], fixture_only=True)
    record = pool["records"][0]
    assert record["annotations"] == [] and record["adjudication"] is None
    assert "support_label" not in record and "confidence" not in record
    assert pool["fixture_only"] is True
    assert sidecar["system_records"][0]["prediction"] == "SUPPORTED"


def test_hypothesis_pool_hides_model_decisions():
    pool, sidecar = generate_annotation_pool("hypothesis", [{
        "split": "train", "query_id": "Q1", "hypothesis": "Combine A and B",
        "evidence": [{"paper_id": "P1"}], "feasibility": "HIGH",
        "accepted": True, "hns": 0.7,
    }])
    record = pool["records"][0]
    assert record["ratings"] == []
    assert not {"model_feasibility", "accepted", "hns"}.intersection(record)
    assert sidecar["system_records"][0]["model_feasibility"] == "HIGH"


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_all_declared_splits_are_preserved(split):
    candidate = contradiction_candidate(split=split)
    pool, _ = generate_annotation_pool("contradiction", [candidate])
    assert pool["records"][0]["split"] == split


def test_invalid_split_is_rejected():
    with pytest.raises(AnnotationPoolError):
        generate_annotation_pool("contradiction", [contradiction_candidate(split="all")])


def test_blinding_validator_rejects_leaked_prediction():
    pool, _ = generate_annotation_pool("contradiction", [contradiction_candidate()])
    pool["records"][0]["confidence"] = 0.9
    with pytest.raises(AnnotationPoolError, match="leaks system fields"):
        validate_annotation_pool(pool, "contradiction")


def test_blinding_validator_rejects_nested_prediction():
    pool, _ = generate_annotation_pool("hypothesis", [{
        "split": "dev", "query_id": "Q1", "hypothesis": "Combine A and B",
        "evidence": [{"paper_id": "P1"}],
    }])
    pool["records"][0]["evidence"][0]["confidence"] = 0.9
    with pytest.raises(AnnotationPoolError, match="leaks system fields"):
        validate_annotation_pool(pool, "hypothesis")


def test_hypothesis_presentation_shuffle_is_reproducible_and_ids_stable():
    candidates = [{
        "split": "dev", "query_id": "Q1", "hypothesis": f"Hypothesis {index}",
        "evidence": [{"paper_id": f"P{index}"}],
    } for index in range(6)]
    sorted_pool, _ = generate_annotation_pool("hypothesis", candidates)
    shuffled, _ = generate_annotation_pool(
        "hypothesis", candidates, presentation_seed="phase-2-fixture"
    )
    repeated, _ = generate_annotation_pool(
        "hypothesis", candidates, presentation_seed="phase-2-fixture"
    )
    sorted_ids = [item["hypothesis_id"] for item in sorted_pool["records"]]
    shuffled_ids = [item["hypothesis_id"] for item in shuffled["records"]]
    assert shuffled_ids != sorted_ids
    assert shuffled_ids == [item["hypothesis_id"] for item in repeated["records"]]
    assert set(shuffled_ids) == set(sorted_ids)
    assert shuffled["presentation_order_randomized"] is True


def test_finalization_rejects_missing_human_labels():
    pool, _ = generate_annotation_pool("contradiction", [contradiction_candidate()])
    with pytest.raises(AnnotationPoolError, match="no human annotations"):
        finalize_annotation_pool(pool)


def test_finalization_requires_adjudication_for_disagreement():
    pool, _ = generate_annotation_pool("contradiction", [contradiction_candidate()])
    pool["records"][0]["annotations"] = [
        {"reviewer_id": "R1", "label": "CONTRADICTION", "reason": "conflict"},
        {"reviewer_id": "R2", "label": "AGREEMENT", "reason": "compatible"},
    ]
    with pytest.raises(AnnotationPoolError, match="disagreement"):
        finalize_annotation_pool(pool)


def test_completed_annotations_finalize_to_valid_benchmark():
    pool, _ = generate_annotation_pool("contradiction", [contradiction_candidate()])
    pool["records"][0]["annotations"] = [
        {"reviewer_id": "R1", "label": "CONTRADICTION", "reason": "conflict"},
        {"reviewer_id": "R2", "label": "CONTRADICTION", "reason": "conflict"},
    ]
    benchmark = finalize_annotation_pool(pool)
    assert benchmark["status"] == "draft"
    assert benchmark["pairs"][0]["label"] == "CONTRADICTION"
    assert benchmark["pairs"][0]["adjudicated"] is False


def test_malformed_annotation_is_rejected():
    pool, _ = generate_annotation_pool("support", [{
        "split": "dev", "query_id": "Q1", "claim": "Claim", "passage_id": "PS1",
        "passage_text": "Evidence", "paper_id": "P1",
    }])
    pool["records"][0]["annotations"] = [{"reviewer_id": "R1", "label": "MAYBE"}]
    with pytest.raises(AnnotationPoolError, match="malformed human annotation"):
        finalize_annotation_pool(pool)


def test_pool_validator_rejects_missing_annotation_evidence():
    pool, _ = generate_annotation_pool("support", [{
        "split": "dev", "query_id": "Q1", "claim": "Claim", "passage_id": "PS1",
        "passage_text": "Evidence", "paper_id": "P1",
    }])
    del pool["records"][0]["passage_text"]
    with pytest.raises(AnnotationPoolError, match="passage_text"):
        validate_annotation_pool(pool, "support")
