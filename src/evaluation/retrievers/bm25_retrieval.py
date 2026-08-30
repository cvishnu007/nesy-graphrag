"""Deterministic BM25 retrieval over the cleaned paper corpus."""

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from src.evaluation.config import BM25_B, BM25_K1
from src.utils.config import (
    CLEAN_FILE,
    TOP_K,
)


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_cached_index = None
_cached_signature = None


def tokenize(text: str) -> list[str]:
    """Lowercase text and keep alphanumeric tokens."""
    return TOKEN_PATTERN.findall(str(text).lower())


class BM25Index:
    """In-memory BM25 index built once and reused for every query."""

    def __init__(
        self,
        papers: Iterable[Mapping],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ):
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")

        self.k1 = float(k1)
        self.b = float(b)
        self.papers = []
        self.document_lengths = []
        self.postings = defaultdict(list)
        self.idf = {}

        seen_ids = set()
        document_frequencies = Counter()

        for raw_paper in papers:
            paper_id = str(raw_paper.get("id", "")).strip()
            if not paper_id:
                raise ValueError("Every BM25 paper must have a non-empty ID")
            if paper_id in seen_ids:
                raise ValueError(f"Duplicate BM25 paper ID: {paper_id}")
            seen_ids.add(paper_id)

            title = str(raw_paper.get("title", "") or "").strip()
            abstract = str(raw_paper.get("abstract", "") or "").strip()
            tokens = tokenize(f"{title} {abstract}")
            term_counts = Counter(tokens)
            document_index = len(self.papers)

            paper = {
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "year": raw_paper.get("year", ""),
                "category": str(
                    raw_paper.get("category", "") or ""
                ),
            }
            self.papers.append(paper)
            self.document_lengths.append(len(tokens))

            for term, frequency in term_counts.items():
                self.postings[term].append(
                    (document_index, frequency)
                )
                document_frequencies[term] += 1

        self.document_count = len(self.papers)
        self.average_document_length = (
            sum(self.document_lengths) / self.document_count
            if self.document_count
            else 0.0
        )

        for term, frequency in document_frequencies.items():
            self.idf[term] = math.log(
                1.0
                + (
                    self.document_count
                    - frequency
                    + 0.5
                )
                / (frequency + 0.5)
            )

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """Return unique papers ordered by descending BM25 score."""
        top_k = int(top_k)

        if top_k <= 0 or self.document_count == 0:
            return []

        query_terms = Counter(tokenize(query))
        if not query_terms:
            return []

        scores = defaultdict(float)
        average_length = self.average_document_length or 1.0

        for term, query_frequency in query_terms.items():
            term_idf = self.idf.get(term)
            if term_idf is None:
                continue

            for document_index, term_frequency in self.postings[term]:
                document_length = self.document_lengths[document_index]
                length_normalization = self.k1 * (
                    1.0
                    - self.b
                    + self.b
                    * document_length
                    / average_length
                )
                term_score = term_idf * (
                    term_frequency
                    * (self.k1 + 1.0)
                    / (term_frequency + length_normalization)
                )
                scores[document_index] += (
                    query_frequency * term_score
                )

        ranked_indices = sorted(
            (
                document_index
                for document_index, score in scores.items()
                if score > 0
            ),
            key=lambda document_index: (
                -scores[document_index],
                self.papers[document_index]["id"],
            ),
        )[:top_k]

        results = []

        for document_index in ranked_indices:
            result = dict(self.papers[document_index])
            result["score"] = round(
                scores[document_index],
                6,
            )
            result["source"] = "bm25"
            results.append(result)

        return results


def _clean_scalar(value) -> str:
    """Convert a dataframe scalar to safe text."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _clean_year(value):
    """Return an integer year when available."""
    if pd.isna(value) or value == "":
        return ""

    try:
        return int(value)
    except (TypeError, ValueError):
        return ""


def load_bm25_index(
    clean_file: str | Path = CLEAN_FILE,
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> BM25Index:
    """Load and validate the cleaned corpus, then build BM25."""
    path = Path(clean_file)
    dataframe = pd.read_json(path)

    required_columns = {
        "id",
        "clean_title",
        "clean_abstract",
        "year",
        "primary_category",
    }
    missing_columns = sorted(
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise RuntimeError(
            f"{path} is missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    papers = [
        {
            "id": _clean_scalar(row["id"]),
            "title": _clean_scalar(row["clean_title"]),
            "abstract": _clean_scalar(row["clean_abstract"]),
            "year": _clean_year(row["year"]),
            "category": _clean_scalar(
                row["primary_category"]
            ),
        }
        for _, row in dataframe.iterrows()
    ]

    return BM25Index(
        papers,
        k1=k1,
        b=b,
    )


def get_bm25_index() -> BM25Index:
    """Build the configured index once per process."""
    global _cached_index, _cached_signature

    path = Path(CLEAN_FILE)
    stat = path.stat()
    signature = (
        str(path.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        BM25_K1,
        BM25_B,
    )

    if _cached_index is None or signature != _cached_signature:
        _cached_index = load_bm25_index(
            path,
            k1=BM25_K1,
            b=BM25_B,
        )
        _cached_signature = signature

    return _cached_index


def bm25_retrieve(
    query: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """Retrieve papers using the cached BM25 index."""
    return get_bm25_index().search(
        query,
        top_k=top_k,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Search the cleaned corpus using BM25"
    )
    parser.add_argument(
        "query",
        help="Research query",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
    )
    arguments = parser.parse_args()

    for rank, paper in enumerate(
        bm25_retrieve(
            arguments.query,
            top_k=arguments.top_k,
        ),
        start=1,
    ):
        print(
            f"{rank:>2}. {paper['score']:.4f} "
            f"{paper['id']} {paper['title']}"
        )
