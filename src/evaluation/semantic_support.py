"""Provider-neutral semantic claim-support verification.

This module deliberately runs after structural provenance validation. It never
interprets the existence of a passage ID as semantic support.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
from typing import Any

from src.evaluation.reasoning_benchmark_io import SUPPORT_LABELS


SupportProvider = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
LABEL_PRIORITY = {
    "CONTRADICTED": 4,
    "SUPPORTED": 3,
    "PARTIALLY_SUPPORTED": 2,
    "UNSUPPORTED": 1,
}


def nli_scores_to_prediction(
    scores: Mapping[str, float], *, partial_entailment_floor: float = 0.20
) -> dict:
    required = {"contradiction", "entailment", "neutral"}
    if set(scores) != required or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or value < 0 or value > 1
        for value in scores.values()
    ):
        raise ValueError("NLI scores must contain three probabilities")
    winning = max(required, key=lambda label: (scores[label], label))
    if winning == "contradiction":
        label = "CONTRADICTED"
    elif winning == "entailment":
        label = "SUPPORTED"
    elif scores["entailment"] >= partial_entailment_floor:
        label = "PARTIALLY_SUPPORTED"
    else:
        label = "UNSUPPORTED"
    return {"label": label, "confidence": float(max(scores.values())), "valid": True}


class LocalNLISupportProvider:
    """Offline claim-support provider backed by a locally cached NLI model."""

    def __init__(self, model_name: str, *, partial_entailment_floor: float = 0.20):
        import torch
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.partial_entailment_floor = partial_entailment_floor
        self.model = CrossEncoder(
            model_name,
            local_files_only=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.id2label = {
            int(index): label.casefold()
            for index, label in self.model.model.config.id2label.items()
        }

    def __call__(self, claim: str, passage: Mapping[str, Any]) -> Mapping[str, Any]:
        probabilities = self.model.predict(
            [(passage.get("text", passage.get("passage_text", "")), claim)],
            show_progress_bar=False,
            apply_softmax=True,
        )[0]
        scores = {
            self.id2label[index]: float(value)
            for index, value in enumerate(probabilities)
        }
        mapped = nli_scores_to_prediction(
            scores, partial_entailment_floor=self.partial_entailment_floor
        )
        return {
            **mapped,
            "reason": "Deterministic mapping from local NLI class probabilities.",
            "model": self.model_name,
            "nli_scores": scores,
        }


@lru_cache(maxsize=2)
def build_local_nli_provider(model_name: str) -> LocalNLISupportProvider:
    return LocalNLISupportProvider(model_name)


def normalize_support_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper().replace(" ", "_")
    return normalized if normalized in SUPPORT_LABELS else None


def parse_support_decision(raw: Any, *, passage_id: str, model: str) -> dict:
    if not isinstance(raw, Mapping):
        return {
            "passage_id": passage_id, "label": "UNSUPPORTED", "confidence": 0.0,
            "valid": False, "model": model, "error": "provider response must be an object",
            "raw": raw,
        }
    label = normalize_support_label(raw.get("label", raw.get("support_label")))
    confidence = raw.get("confidence")
    confidence_valid = (
        isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
        and 0 <= confidence <= 1
    )
    valid = label is not None and confidence_valid
    return {
        "passage_id": passage_id,
        "label": label or "UNSUPPORTED",
        "confidence": float(confidence) if confidence_valid else 0.0,
        "valid": valid,
        "model": str(raw.get("model") or model),
        "reason": str(raw.get("reason") or ""),
        "error": "" if valid else "malformed provider decision",
        "raw": dict(raw),
    }


def aggregate_passage_decisions(decisions: Iterable[dict], *, min_confidence: float = 0.70) -> dict:
    if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)) or not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    rows = list(decisions)
    eligible = [row for row in rows if row.get("valid") and row.get("confidence", 0) >= min_confidence]
    if not eligible:
        return {
            "label": "UNSUPPORTED", "confidence": 0.0, "valid": False,
            "error": "no valid decision met the confidence threshold",
            "eligible_passage_count": 0,
        }
    # A confident contradiction dominates. Otherwise any direct support is
    # sufficient; partial support dominates merely related/unsupported text.
    winning = max(eligible, key=lambda row: (LABEL_PRIORITY[row["label"]], row["confidence"], row["passage_id"]))
    return {
        "label": winning["label"],
        "confidence": max(row["confidence"] for row in eligible if row["label"] == winning["label"]),
        "valid": True,
        "error": "",
        "eligible_passage_count": len(eligible),
    }


def verify_claim_support(
    claim: str,
    passages: Iterable[Mapping[str, Any]],
    *,
    provider: SupportProvider | None,
    claim_id: str = "",
    model: str = "unconfigured",
    min_confidence: float = 0.70,
) -> dict:
    """Verify a claim against structurally validated passages.

    ``provider`` is called once per passage and may be a local NLI model or a
    deterministic structured LLM adapter. No provider is invoked by default.
    """
    if not isinstance(claim, str) or not claim.strip():
        return {
            "claim_id": claim_id, "passage_ids": [], "support_label": "UNSUPPORTED",
            "confidence": 0.0, "per_passage": [], "valid": False, "model": model,
            "audit": {"errors": ["claim must be non-empty"]},
        }
    materialized = list(passages)
    passage_ids = []
    errors = []
    for index, passage in enumerate(materialized):
        if not isinstance(passage, Mapping):
            errors.append(f"passage[{index}] must be an object")
            continue
        passage_id = passage.get("id", passage.get("passage_id"))
        text = passage.get("text", passage.get("passage_text"))
        if not isinstance(passage_id, str) or not passage_id.strip() or not isinstance(text, str) or not text.strip():
            errors.append(f"passage[{index}] requires non-empty id and text")
            continue
        if passage_id in passage_ids:
            errors.append(f"duplicate passage ID: {passage_id}")
            continue
        passage_ids.append(passage_id)
    if errors or not passage_ids or provider is None:
        if not passage_ids:
            errors.append("no valid evidence passages")
        if provider is None:
            errors.append("semantic support provider is unavailable")
        return {
            "claim_id": claim_id, "passage_ids": passage_ids,
            "support_label": "UNSUPPORTED", "confidence": 0.0,
            "per_passage": [], "valid": False, "model": model,
            "audit": {"errors": list(dict.fromkeys(errors))},
        }

    decisions = []
    for passage in materialized:
        passage_id = passage.get("id", passage.get("passage_id"))
        if passage_id not in passage_ids:
            continue
        try:
            raw = provider(claim.strip(), passage)
        except Exception as error:  # provider failure is data, not a fabricated decision
            raw = {"error": type(error).__name__}
        decisions.append(parse_support_decision(raw, passage_id=passage_id, model=model))
    aggregate = aggregate_passage_decisions(decisions, min_confidence=min_confidence)
    audit_errors = [row["error"] for row in decisions if row.get("error")]
    if aggregate["error"]:
        audit_errors.append(aggregate["error"])
    return {
        "claim_id": claim_id,
        "passage_ids": passage_ids,
        "support_label": aggregate["label"],
        "confidence": aggregate["confidence"],
        "per_passage": decisions,
        "valid": aggregate["valid"],
        "model": model,
        "audit": {
            "errors": audit_errors,
            "min_confidence": min_confidence,
            "eligible_passage_count": aggregate["eligible_passage_count"],
        },
    }
