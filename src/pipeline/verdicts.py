"""
Shared parsing helpers for LLM contradiction verdicts.
"""


VALID_VERDICTS = {"CONTRADICTION", "AGREEMENT", "DIFFERENT SCOPE"}


def extract_verdict(analysis: str) -> str:
    """Extract the explicit VERDICT line from an LLM analysis block."""
    for line in (analysis or "").splitlines():
        normalized = line.strip().lstrip("-* ").strip().upper()
        if not normalized.startswith("VERDICT:"):
            continue

        verdict = normalized.split(":", 1)[1].strip()
        verdict = verdict.replace("[", "").replace("]", "").strip(" .")
        return verdict if verdict in VALID_VERDICTS else "UNKNOWN"

    return "UNKNOWN"


def has_contradiction_verdict(analysis: str) -> bool:
    return extract_verdict(analysis) == "CONTRADICTION"
