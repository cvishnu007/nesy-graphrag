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


def parse_contradiction_response(text):
    """Parse strict verdict fields without relying on keyword presence."""
    response = text or ""
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
