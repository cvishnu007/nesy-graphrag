import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import (
    LLM_MODEL, LLM_MODEL_FALLBACK, GROQ_MAX_RETRIES, TOP_K,
    SEMANTIC_SUPPORT_MIN_CONFIDENCE, SEMANTIC_SUPPORT_MODEL,
)
from src.pipeline.retrieval import nesy_retrieve, vector_only_retrieve
from src.pipeline.validator import validate_citations
from src.pipeline.prompts import build_review_prompt, build_review_repair_prompt
from src.pipeline.provenance import (
    build_passages,
    format_passage_context,
    parse_review_claims,
    render_grounded_review,
    validate_claim_provenance,
)
from src.evaluation.semantic_support import build_local_nli_provider, verify_claim_support
from src.utils.groq_client import groq_chat_with_retry


def _console_safe(text):
    encoding = sys.stdout.encoding or "utf-8"
    return str(text).encode(encoding, errors="replace").decode(encoding)


def _apply_semantic_support(
    claims,
    passages,
    *,
    provider,
    model,
    min_confidence,
):
    """Evaluate structurally grounded claims without discarding audit records."""
    passage_index = {passage["id"]: passage for passage in passages}
    accepted = []
    rejected = []
    decisions = []
    for index, original in enumerate(claims, 1):
        claim = dict(original)
        evidence = [
            passage_index[passage_id]
            for passage_id in claim.get("cited_passage_ids", [])
            if passage_id in passage_index
        ]
        decision = verify_claim_support(
            claim.get("text", ""),
            evidence,
            provider=provider,
            claim_id=f"claim-{index:03d}",
            model=model,
            min_confidence=min_confidence,
        )
        claim["semantic_support"] = decision
        decisions.append(decision)
        if decision["valid"] and decision["support_label"] in {
            "SUPPORTED", "PARTIALLY_SUPPORTED"
        }:
            accepted.append(claim)
        else:
            rejected.append(claim)
    return accepted, rejected, {
        "enabled": True,
        "annotation_source": "AI-generated semantic support decisions",
        "human_review": False,
        "model": model,
        "min_confidence": min_confidence,
        "decisions": decisions,
        "stats": {
            "evaluated_claims": len(claims),
            "accepted_claims": len(accepted),
            "rejected_claims": len(rejected),
        },
    }


def llm_review(
    groq_client,
    driver,
    query,
    top_k=TOP_K,
    *,
    baseline=False,
    support_provider=None,
    support_model=SEMANTIC_SUPPORT_MODEL,
    support_min_confidence=SEMANTIC_SUPPORT_MIN_CONFIDENCE,
):
    """Generate a literature review via NeSy retrieval + LLM synthesis.

    Parameters
    ----------
    groq_client : groq.Groq instance
    driver      : Neo4j driver
    query       : str — research query
    top_k       : int — papers to retrieve
    baseline    : bool — if True, use vector-only retrieval (no graph expansion)
    """
    mode_label = "BASELINE" if baseline else "REVIEW"
    print(f"\n{'='*60}")
    print(f"[{mode_label} MODE] {query}")
    print(f"{'='*60}")

    if baseline:
        papers = vector_only_retrieve(query, top_k=top_k)
    else:
        papers = nesy_retrieve(driver, query, top_k=top_k)
    print(f"[{'Baseline' if baseline else 'NeSy'}] Retrieved {len(papers)} papers")

    verified = validate_citations(driver, [p["id"] for p in papers if p.get("id")])
    print(f"[Validator] {len(verified)}/{len(papers)} citations verified")

    passages = build_passages(papers, verified)
    prompt = build_review_prompt(format_passage_context(passages), query)

    print("[LLM] Generating answer...")
    try:
        raw_answer = groq_chat_with_retry(
            groq_client, prompt,
            model=LLM_MODEL,
            fallback_model=LLM_MODEL_FALLBACK,
            max_tokens=1400,
            temperature=0.0,
            max_retries=GROQ_MAX_RETRIES,
        )
    except Exception as e:
        raw_answer = f"LLM call failed: {e}"

    parsed_claims, parse_errors = parse_review_claims(raw_answer)
    provenance = validate_claim_provenance(
        parsed_claims,
        passages,
        parse_errors=parse_errors,
    )
    raw_answers = [raw_answer]
    if (
        not provenance["claims"]
        and passages
        and not raw_answer.startswith("LLM call failed:")
    ):
        print("[Provenance] No valid claims; attempting one structured-output repair...")
        repair_prompt = build_review_repair_prompt(prompt, raw_answer, parse_errors)
        try:
            repaired_answer = groq_chat_with_retry(
                groq_client,
                repair_prompt,
                model=LLM_MODEL,
                fallback_model=LLM_MODEL_FALLBACK,
                max_tokens=1400,
                temperature=0.0,
                max_retries=GROQ_MAX_RETRIES,
            )
            raw_answers.append(repaired_answer)
            repaired_claims, repair_errors = parse_review_claims(repaired_answer)
            provenance = validate_claim_provenance(
                repaired_claims,
                passages,
                parse_errors=repair_errors,
            )
            raw_answer = repaired_answer
        except Exception as exc:
            provenance["parse_errors"].append(f"repair call failed: {exc}")

    provenance["stats"]["generation_attempts"] = len(raw_answers)
    semantic_unsupported_claims = []
    semantic_support = {"enabled": False}
    output_claims = provenance["claims"]
    provider_error = None
    semantic_requested = support_provider is not None or support_model != "unconfigured"
    if support_provider is None and semantic_requested:
        try:
            support_provider = build_local_nli_provider(support_model)
        except Exception as error:
            provider_error = f"{type(error).__name__}: {error}"
    if semantic_requested:
        output_claims, semantic_unsupported_claims, semantic_support = (
            _apply_semantic_support(
                provenance["claims"],
                passages,
                provider=support_provider,
                model=support_model,
                min_confidence=support_min_confidence,
            )
        )
        if provider_error:
            semantic_support["provider_initialization_error"] = provider_error
        provenance["semantic_support"] = semantic_support
    answer = render_grounded_review(output_claims)
    stats = provenance["stats"]
    print(
        "[Provenance] "
        f"{stats['grounded_claims']}/{stats['total_claims']} claims grounded; "
        f"{stats['invalid_citations']} invalid passage citations blocked"
    )

    print(f"\n{'-'*60}")
    print("ANSWER:")
    print(f"{'-'*60}")
    print(_console_safe(answer))

    return {
        "query": query,
        "papers": papers,
        "answer": answer,
        "raw_answer": raw_answer,
        "raw_answers": raw_answers,
        "verified": verified,
        "passages": passages,
        "claims": output_claims,
        "unsupported_claims": provenance["unsupported_claims"],
        "semantic_unsupported_claims": semantic_unsupported_claims,
        "semantic_support": semantic_support,
        "provenance": provenance,
    }
