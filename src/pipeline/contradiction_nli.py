"""Sentence-level, NLI-scored contradiction candidates with evidence provenance.

The concept-overlap generator in ``contradiction.py`` pairs papers that share
noun phrases. On the current graph those pairs came from generic hub phrases:
with generic concepts excluded it finds no candidates for any of 26 probe
queries. This generator instead compares *result-bearing sentences* across
retrieved papers:

1. split each abstract into sentences (the same splitter used for review
   provenance) and keep sentences that report findings;
2. keep cross-paper sentence pairs that discuss the same thing (embedding
   cosine >= ``min_topic_similarity``);
3. score each pair with a local NLI cross-encoder in both directions and use
   the mean contradiction probability (symmetric, so pair order is irrelevant);
4. rank paper pairs by their strongest sentence pair and return those two
   sentences as evidence.

The NLI probability is a screening score, not a verdict: candidates still go
to the structured LLM verdict, and the score is not calibrated for this domain.
"""
from __future__ import annotations

import re
from functools import lru_cache
from itertools import combinations

import numpy as np

from src.pipeline.provenance import passage_id, split_abstract_sentences

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
NLI_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"
TOPIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Sentences that report findings rather than motivation or method setup.
_FINDING = re.compile(
    r"\b(show|shows|showed|shown|demonstrate[sd]?|find|finds|found|observe[sd]?|reveal[sd]?|"
    r"outperform[s]?|outperformed|improve[sd]?|improvement|achieve[sd]?|reduce[sd]?|increase[sd]?|"
    r"decrease[sd]?|fail[s]?|failed|cannot|does not|do not|no significant|significant(ly)?|"
    r"better|worse|superior|inferior|comparable|effective|ineffective|accuracy|results? (show|indicate|suggest))\b",
    re.IGNORECASE,
)


# The finding must be the authors' own (first person or reported results),
# not a statement about the field.
_AUTHORIAL = re.compile(
    r"\b(we|our|this (paper|work|study)|results|experiments?|empirical(ly)?|evaluation|analysis|findings)\b",
    re.IGNORECASE,
)
# Field-level background that reads like a claim but reports no new finding.
_BACKGROUND = re.compile(
    r"^(in recent years|recently|nowadays|over the (past|last)|with the (rapid|recent)|"
    r"[\w\s\-()]{0,60}\b(have|has) (achieved|attracted|become|been (widely|increasingly|successfully)|shown (great|remarkable|promising)))",
    re.IGNORECASE,
)


def is_finding(sentence: str) -> bool:
    """True for sentences reporting the authors' own results."""
    return (
        len(sentence.split()) >= 6
        and bool(_FINDING.search(sentence))
        and bool(_AUTHORIAL.search(sentence))
        and not _BACKGROUND.search(sentence.strip())
    )


def finding_sentences(paper: dict, *, max_sentences: int = 6) -> list[dict]:
    """Return result-bearing sentences with stable passage IDs."""
    rows = []
    for index, sentence in enumerate(split_abstract_sentences(paper.get("abstract") or ""), start=1):
        if not is_finding(sentence):
            continue
        rows.append({"id": passage_id(paper["id"], index), "paper_id": paper["id"],
                     "sentence_index": index, "text": sentence})
    return rows[:max_sentences]


@lru_cache(maxsize=1)
def _nli():
    import torch
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(NLI_MODEL, revision=NLI_REVISION, local_files_only=True,
                         device="cuda" if torch.cuda.is_available() else "cpu")
    labels = {int(i): label.casefold() for i, label in model.model.config.id2label.items()}
    return model, next(i for i, label in labels.items() if label.startswith("contradict"))


@lru_cache(maxsize=1)
def _topic_encoder():
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(TOPIC_MODEL, local_files_only=True,
                               device="cuda" if torch.cuda.is_available() else "cpu")


def contradiction_probabilities(pairs: list[tuple[str, str]], *, scorer=None) -> np.ndarray:
    """Mean contradiction probability over both sentence orders."""
    if not pairs:
        return np.zeros(0)
    if scorer is not None:
        return np.asarray(scorer(pairs), dtype=float)
    model, contradiction_index = _nli()
    forward = model.predict(pairs, apply_softmax=True, show_progress_bar=False, batch_size=32)
    backward = model.predict([(b, a) for a, b in pairs], apply_softmax=True, show_progress_bar=False, batch_size=32)
    return (np.asarray(forward)[:, contradiction_index] + np.asarray(backward)[:, contradiction_index]) / 2


def rank_contradiction_candidates(
    papers: list[dict],
    *,
    top_k: int = 5,
    min_topic_similarity: float = 0.5,
    min_contradiction: float = 0.0,
    embed=None,
    scorer=None,
) -> list[dict]:
    """Rank cross-paper pairs by their strongest contradicting finding pair."""
    sentences = {paper["id"]: finding_sentences(paper) for paper in papers if paper.get("id")}
    flat = [row for rows in sentences.values() for row in rows]
    if len(sentences) < 2 or not flat:
        return []
    if embed is None:
        vectors = _topic_encoder().encode([row["text"] for row in flat], normalize_embeddings=True,
                                          show_progress_bar=False)
    else:
        vectors = np.asarray(embed([row["text"] for row in flat]), dtype=float)
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    position = {row["id"]: index for index, row in enumerate(flat)}
    by_id = {paper["id"]: paper for paper in papers}

    candidate_pairs, sentence_pairs = [], []
    from src.pipeline.near_duplicates import are_near_duplicates

    for first, second in combinations(sorted(sentences), 2):
        if are_near_duplicates(first, second):
            continue  # near-identical abstracts can only look like agreement
        for a in sentences[first]:
            for b in sentences[second]:
                similarity = float(vectors[position[a["id"]]] @ vectors[position[b["id"]]])
                if similarity >= min_topic_similarity:
                    candidate_pairs.append((first, second, a, b, similarity))
                    sentence_pairs.append((a["text"], b["text"]))
    scores = contradiction_probabilities(sentence_pairs, scorer=scorer)

    best = {}
    for (first, second, a, b, similarity), score in zip(candidate_pairs, scores):
        key = (first, second)
        if score >= min_contradiction and (key not in best or score > best[key]["nli_contradiction"]):
            best[key] = {"evidence_1": a, "evidence_2": b, "topic_similarity": round(similarity, 4),
                         "nli_contradiction": round(float(score), 4)}
    ranked = sorted(best.items(), key=lambda item: (-item[1]["nli_contradiction"], item[0]))
    output = []
    for (first, second), evidence in ranked[:top_k]:
        paper1, paper2 = by_id[first], by_id[second]
        output.append({
            "paper1": {key: paper1.get(key) for key in ("id", "title", "abstract", "year")},
            "paper2": {key: paper2.get(key) for key in ("id", "title", "abstract", "year")},
            "candidate_score": evidence["nli_contradiction"],
            "candidate_method": "nli_sentence_pairs",
            "compared_sentence_pairs": sum(1 for c in candidate_pairs if (c[0], c[1]) == (first, second)),
            **evidence,
            "flag": f"Finding sentences with NLI contradiction {evidence['nli_contradiction']:.2f}",
        })
    return output


def detect_contradictions_nli(driver, query, top_k=5, *, retrieve_k=10, **options):
    """Retrieve papers for a query, then rank NLI-scored contradiction candidates."""
    from src.pipeline.retrieval import nesy_retrieve

    papers = nesy_retrieve(driver, query, top_k=retrieve_k)
    return rank_contradiction_candidates(papers, top_k=top_k, **options)
