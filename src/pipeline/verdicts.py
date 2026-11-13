import json
import re


VALID_CONTRADICTION_VERDICTS = {
    "CONTRADICTION",
    "AGREEMENT",
    "DIFFERENT SCOPE",
}

_VERDICT_PATTERN = re.compile(
    r"^\s*VERDICT\s*:\s*(CONTRADICTION|AGREEMENT|DIFFERENT\s+SCOPE)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CONFIDENCE_PATTERN = re.compile(
    r"^\s*CONFIDENCE\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_REASON_PATTERN = re.compile(
    r"^\s*REASON\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


CONTRADICTION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "confidence", "reason", "claim_1", "claim_2"],
    "properties": {
        "verdict": {"type": "string", "enum": sorted(VALID_CONTRADICTION_VERDICTS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "claim_1": {"type": "string"},
        "claim_2": {"type": "string"},
    },
}


def _parse_json_verdict(response):
    """Return parsed fields from a schema-constrained JSON verdict, or None."""
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    verdict = re.sub(r"\s+", " ", str(payload.get("verdict", "")).upper().strip())
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        confidence = None
    valid = verdict in VALID_CONTRADICTION_VERDICTS
    return {
        "verdict": verdict if valid else "UNKNOWN",
        "confidence": float(confidence) if confidence is not None else None,
        "reason": str(payload.get("reason") or "").strip(),
        "valid": valid,
    }


def parse_contradiction_response(text):
    """Parse strict verdict fields from JSON or the legacy line format."""
    response = text or ""
    structured = _parse_json_verdict(response)
    if structured is not None:
        return structured
    verdict_match = _VERDICT_PATTERN.search(response)
    confidence_match = _CONFIDENCE_PATTERN.search(response)
    reason_match = _REASON_PATTERN.search(response)

    verdict = (
        re.sub(r"\s+", " ", verdict_match.group(1).upper())
        if verdict_match
        else "UNKNOWN"
    )
    confidence = float(confidence_match.group(1)) if confidence_match else None
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason": reason_match.group(1).strip() if reason_match else "",
        "valid": verdict in VALID_CONTRADICTION_VERDICTS,
    }


def contradiction_verdict(item):
    """Return a normalized verdict from structured or legacy result items."""
    verdict = str(item.get("verdict", "")).upper().strip()
    if verdict in VALID_CONTRADICTION_VERDICTS:
        return verdict
    return parse_contradiction_response(item.get("llm_analysis", ""))["verdict"]


def is_confident_contradiction(item, min_confidence):
    """Require both an exact contradiction verdict and a confidence threshold."""
    parsed = parse_contradiction_response(item.get("llm_analysis", ""))
    confidence = item.get("confidence", parsed["confidence"])
    return (
        contradiction_verdict(item) == "CONTRADICTION"
        and confidence is not None
        and float(confidence) >= min_confidence
    )
