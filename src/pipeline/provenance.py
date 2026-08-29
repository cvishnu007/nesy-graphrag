"""Deterministic claim-to-passage provenance for generated reviews."""

import hashlib
import re
from typing import Iterable

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_FIELD_LINE = re.compile(r"^(CLAIM|EVIDENCE)\s*:\s*(.*)$", re.IGNORECASE)
_NON_TERMINAL_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "et al.",
    "fig.",
    "eq.",
    "ref.",
    "vs.",
)


def split_abstract_sentences(abstract: str) -> list[str]:
    """Split an abstract into stable, non-empty sentence passages."""
    normalized = " ".join((abstract or "").split())
    if not normalized:
        return []
    passages = []
    for candidate in _SENTENCE_BOUNDARY.split(normalized):
        candidate = candidate.strip()
        if not candidate:
            continue
        if passages and passages[-1].lower().endswith(_NON_TERMINAL_ABBREVIATIONS):
            passages[-1] = f"{passages[-1]} {candidate}"
        else:
            passages.append(candidate)
    return passages


def passage_id(paper_id: str, sentence_index: int) -> str:
    """Return a compact stable ID without exposing source-ID punctuation."""
    digest = hashlib.sha256(str(paper_id).encode("utf-8")).hexdigest()[:16]
    return f"P{digest}-S{sentence_index:03d}"


def build_passages(papers: Iterable[dict], verified: dict[str, str]) -> list[dict]:
    """Create sentence passages only for papers already verified in Neo4j."""
    passages = []
    for paper in papers:
        paper_id_value = str(paper.get("id") or "")
        if not paper_id_value or paper_id_value not in verified:
            continue
        sentences = split_abstract_sentences(paper.get("abstract") or "")
        for index, sentence in enumerate(sentences, start=1):
            passages.append(
                {
                    "id": passage_id(paper_id_value, index),
                    "paper_id": paper_id_value,
                    "paper_title": verified[paper_id_value],
                    "sentence_index": index,
                    "text": sentence,
                    "year": paper.get("year"),
                    "category": paper.get("category"),
                }
            )
    return passages


def format_passage_context(passages: Iterable[dict]) -> str:
    """Format passages for an LLM while retaining exact citation IDs."""
    rows = []
    for item in passages:
        rows.append(
            f"[{item['id']}] paper_id={item['paper_id']} | "
            f"title={item['paper_title']} | year={item.get('year')} | "
            f"text={item['text']}"
        )
    return "\n".join(rows)


def parse_review_claims(response: str) -> tuple[list[dict], list[str]]:
    """Parse repeated CLAIM/EVIDENCE blocks from strict plain-text output."""
    claims = []
    errors = []
    current = None
    cleaned = (response or "").replace("```text", "").replace("```", "")

    def finish_current():
        nonlocal current
        if current is None:
            return
        if not current["text"]:
            errors.append("empty claim")
        claims.append(current)
        current = None

    for raw_line in cleaned.splitlines():
        line = raw_line.strip().replace("**", "")
        if not line:
            continue
        match = _FIELD_LINE.match(line)
        if not match:
            errors.append(f"unrecognized line: {line[:80]}")
            continue
        field, value = match.groups()
        if field.upper() == "CLAIM":
            finish_current()
            current = {"text": value.strip(), "cited_passage_ids": []}
            continue
        if current is None:
            errors.append("evidence appeared before a claim")
            continue
        current["cited_passage_ids"].extend(
            token.strip().strip("[]")
            for token in value.split(",")
            if token.strip().strip("[]")
        )

    finish_current()
    if not claims:
        errors.append("no claim blocks found")
    return claims, errors


def validate_claim_provenance(
    claims: Iterable[dict],
    passages: Iterable[dict],
    *,
    parse_errors: Iterable[str] = (),
) -> dict:
    """Validate every cited passage ID and partition claims for display/audit."""
    passage_index = {item["id"]: item for item in passages}
    grounded = []
    unsupported = []
    total_references = 0
    valid_references = 0

    for claim in claims:
        cited_ids = list(dict.fromkeys(claim.get("cited_passage_ids") or []))
        valid_ids = [item_id for item_id in cited_ids if item_id in passage_index]
        invalid_ids = [item_id for item_id in cited_ids if item_id not in passage_index]
        total_references += len(cited_ids)
        valid_references += len(valid_ids)
        validated = {
            "text": (claim.get("text") or "").strip(),
            "cited_passage_ids": cited_ids,
            "evidence": [passage_index[item_id] for item_id in valid_ids],
            "invalid_passage_ids": invalid_ids,
        }
        reasons = []
        if not validated["text"]:
            reasons.append("empty_claim")
        if not cited_ids:
            reasons.append("missing_evidence")
        if invalid_ids:
            reasons.append("invalid_passage_ids")
        validated["grounded"] = not reasons
        validated["rejection_reasons"] = reasons
        (grounded if validated["grounded"] else unsupported).append(validated)

    total_claims = len(grounded) + len(unsupported)
    citation_precision = valid_references / total_references if total_references else 0.0
    claim_coverage = len(grounded) / total_claims if total_claims else 0.0
    parse_errors = list(parse_errors)
    return {
        "claims": grounded,
        "unsupported_claims": unsupported,
        "stats": {
            "total_claims": total_claims,
            "grounded_claims": len(grounded),
            "unsupported_claims": len(unsupported),
            "total_citations": total_references,
            "valid_citations": valid_references,
            "invalid_citations": total_references - valid_references,
            "citation_precision": round(citation_precision, 4),
            "claim_coverage": round(claim_coverage, 4),
            "valid_output": total_claims > 0 and not unsupported and not parse_errors,
        },
        "parse_errors": parse_errors,
    }


def render_grounded_review(claims: Iterable[dict]) -> str:
    """Render only claims whose complete passage citation set is valid."""
    lines = []
    for claim in claims:
        references = ", ".join(
            f"[{item['id']}]" for item in claim.get("evidence", [])
        )
        lines.append(f"- {claim['text']} {references}".rstrip())
    if not lines:
        return "No claim-level grounded review could be produced."
    return "\n\n".join(lines)
