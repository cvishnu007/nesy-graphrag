import pytest

from src.pipeline import review
from src.pipeline.metrics import compute_provenance_ts
from src.pipeline.provenance import (
    build_passages,
    format_passage_context,
    parse_review_claims,
    passage_id,
    render_grounded_review,
    split_abstract_sentences,
    validate_claim_provenance,
)


@pytest.fixture
def verified_papers():
    papers = [
        {
            "id": "paper:1",
            "title": "Verified Paper",
            "abstract": "First finding is supported. Second finding extends it.",
            "year": 2024,
            "category": "Computer Science",
            "source": "neural",
            "score": 0.9,
        },
        {
            "id": "paper:2",
            "title": "Unverified Paper",
            "abstract": "This content must not reach the prompt.",
            "year": 2023,
            "category": "Computer Science",
            "source": "symbolic",
            "score": 0.8,
        },
    ]
    return papers, {"paper:1": "Verified Paper"}


def test_passage_ids_are_stable_and_verified_papers_only(verified_papers):
    papers, verified = verified_papers

    first = build_passages(papers, verified)
    second = build_passages(papers, verified)

    assert first == second
    assert [item["id"] for item in first] == [
        passage_id("paper:1", 1),
        passage_id("paper:1", 2),
    ]
    assert {item["paper_id"] for item in first} == {"paper:1"}
    assert "Unverified Paper" not in format_passage_context(first)


def test_sentence_splitter_handles_lowercase_and_scientific_abbreviations():
    sentences = split_abstract_sentences(
        "first result holds. second uses prior work, e.g. graph methods. third follows."
    )

    assert sentences == [
        "first result holds.",
        "second uses prior work, e.g. graph methods.",
        "third follows.",
    ]


def test_fabricated_passage_id_rejects_the_complete_claim(verified_papers):
    papers, verified = verified_papers
    passages = build_passages(papers, verified)
    real_id = passages[0]["id"]
    response = (
        f"CLAIM: A grounded claim.\nEVIDENCE: [{real_id}]\n"
        f"CLAIM: A mixed claim.\nEVIDENCE: [{real_id}], [PFAKE-S001]"
    )

    parsed, errors = parse_review_claims(response)
    provenance = validate_claim_provenance(parsed, passages, parse_errors=errors)

    assert errors == []
    assert [claim["text"] for claim in provenance["claims"]] == [
        "A grounded claim."
    ]
    assert provenance["unsupported_claims"][0]["rejection_reasons"] == [
        "invalid_passage_ids"
    ]
    assert provenance["stats"]["invalid_citations"] == 1
    assert "A mixed claim" not in render_grounded_review(provenance["claims"])


@pytest.mark.parametrize(
    "response",
    [
        "",
        "An answer without structured evidence.",
        "EVIDENCE: [PFAKE-S001]",
    ],
)
def test_malformed_responses_produce_no_grounded_claims(response):
    claims, errors = parse_review_claims(response)
    provenance = validate_claim_provenance(claims, [], parse_errors=errors)

    assert provenance["claims"] == []
    assert provenance["stats"]["valid_output"] is False
    assert errors


def test_provenance_ts_combines_citation_integrity_and_claim_coverage():
    provenance = {
        "stats": {
            "total_claims": 2,
            "grounded_claims": 1,
            "total_citations": 3,
            "valid_citations": 2,
        }
    }

    result = compute_provenance_ts(provenance)

    assert result["citation_integrity"] == pytest.approx(0.6667)
    assert result["claim_coverage"] == 0.5
    assert result["ts"] == pytest.approx(0.5833)


def test_review_returns_only_claims_with_valid_passage_ids(
    monkeypatch,
    verified_papers,
):
    papers, verified = verified_papers
    real_id = passage_id("paper:1", 1)
    raw_answer = (
        f"CLAIM: Supported by the abstract.\nEVIDENCE: [{real_id}]\n"
        "CLAIM: Fabricated support.\nEVIDENCE: [PFAKE-S001]"
    )
    captured_prompt = {}

    monkeypatch.setattr(review, "nesy_retrieve", lambda *args, **kwargs: papers)
    monkeypatch.setattr(review, "validate_citations", lambda *args, **kwargs: verified)

    def fake_chat(client, prompt, **kwargs):
        captured_prompt["value"] = prompt
        return raw_answer

    monkeypatch.setattr(review, "groq_chat_with_retry", fake_chat)

    result = review.llm_review(object(), object(), "test query", top_k=2)

    assert real_id in captured_prompt["value"]
    assert "Unverified Paper" not in captured_prompt["value"]
    assert [claim["text"] for claim in result["claims"]] == [
        "Supported by the abstract."
    ]
    assert len(result["unsupported_claims"]) == 1
    assert "Fabricated support" not in result["answer"]
    assert result["provenance"]["stats"]["generation_attempts"] == 1


def test_review_repairs_once_when_first_response_has_no_valid_claims(
    monkeypatch,
    verified_papers,
):
    papers, verified = verified_papers
    real_id = passage_id("paper:1", 1)
    responses = iter(
        [
            "An unstructured response.",
            f"CLAIM: Repaired claim.\nEVIDENCE: [{real_id}]",
        ]
    )

    monkeypatch.setattr(review, "nesy_retrieve", lambda *args, **kwargs: papers)
    monkeypatch.setattr(review, "validate_citations", lambda *args, **kwargs: verified)
    monkeypatch.setattr(
        review,
        "groq_chat_with_retry",
        lambda *args, **kwargs: next(responses),
    )

    result = review.llm_review(object(), object(), "test query", top_k=2)

    assert [claim["text"] for claim in result["claims"]] == ["Repaired claim."]
    assert len(result["raw_answers"]) == 2
    assert result["provenance"]["stats"]["generation_attempts"] == 2
