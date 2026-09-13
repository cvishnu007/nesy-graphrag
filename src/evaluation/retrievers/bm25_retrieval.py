"""Deterministic BM25 over the same title/abstract corpus as neural retrieval."""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.casefold())


class BM25Retriever:
    def __init__(self, papers, *, k1: float = 1.5, b: float = 0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError('BM25 requires finite k1 > 0 and b in [0, 1]')
        self.k1, self.b = k1, b
        self.papers = {}
        self.lengths = {}
        self.postings = defaultdict(dict)
        for paper in papers:
            paper_id = str(paper.get('id', '')).strip()
            if not paper_id or paper_id in self.papers:
                raise ValueError('blank or duplicate paper ID')
            title = paper.get('clean_title', paper.get('title', '')) or ''
            abstract = paper.get('clean_abstract', paper.get('abstract', '')) or ''
            self.papers[paper_id] = {
                'id': paper_id, 'title': title, 'abstract': abstract,
                'year': paper.get('year', 0),
                'category': paper.get('primary_category', paper.get('category', 'unknown')),
                'source': 'bm25',
            }
            terms = Counter(tokenize(f'{title} {abstract}'))
            self.lengths[paper_id] = sum(terms.values())
            for term, frequency in terms.items():
                self.postings[term][paper_id] = frequency
        self.average_length = sum(self.lengths.values()) / max(len(self.papers), 1)

    @classmethod
    def from_json(cls, path: str | Path, **kwargs):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')), **kwargs)

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError('top_k must be a positive integer')
        scores = defaultdict(float)
        for term in sorted(set(tokenize(query))):
            postings = self.postings.get(term, {})
            df, count = len(postings), len(self.papers)
            idf = math.log(1 + (count - df + 0.5) / (df + 0.5))
            for paper_id, frequency in postings.items():
                norm = 1 - self.b + self.b * self.lengths[paper_id] / (self.average_length or 1)
                scores[paper_id] += idf * frequency * (self.k1 + 1) / (frequency + self.k1 * norm)
        ranked = sorted(scores, key=lambda pid: (-scores[pid], pid))[:top_k]
        return [{**self.papers[pid], 'score': scores[pid]} for pid in ranked]
