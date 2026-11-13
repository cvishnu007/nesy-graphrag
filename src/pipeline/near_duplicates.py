"""Collapse near-identical abstracts in ranked results without deleting data.

Pairs are abstract embeddings with cosine >= 0.995 (``evaluation/near_duplicate_pairs.json``).
They are usually preprint/published versions of one paper, but can also be
consecutive editions of a report, so they are treated as redundant results,
never merged or removed from the stores.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from src.utils.config import NEAR_DUPLICATE_FILE


@lru_cache(maxsize=4)
def near_duplicate_map(path: str = NEAR_DUPLICATE_FILE) -> dict[str, frozenset]:
    file = Path(path)
    if not file.exists():
        return {}
    mapping: dict[str, set] = {}
    for first, second, _score in json.loads(file.read_text(encoding="utf-8"))["pairs"]:
        mapping.setdefault(first, set()).add(second)
        mapping.setdefault(second, set()).add(first)
    return {key: frozenset(value) for key, value in mapping.items()}


def collapse(papers: list[dict], mapping: dict | None = None) -> list[dict]:
    """Keep the first (highest-ranked) paper of each near-duplicate group."""
    mapping = near_duplicate_map() if mapping is None else mapping
    kept, seen = [], set()
    for paper in papers:
        if paper["id"] in seen:
            continue
        kept.append(paper)
        seen.add(paper["id"])
        seen |= mapping.get(paper["id"], frozenset())
    return kept


def are_near_duplicates(first: str, second: str, mapping: dict | None = None) -> bool:
    mapping = near_duplicate_map() if mapping is None else mapping
    return second in mapping.get(first, frozenset())
