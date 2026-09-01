"""Flatten native pipeline outputs into unlabeled Phase 2 candidate records."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from src.pipeline.hypothesis import parse_hypothesis_response


class CandidateExportError(ValueError):
    """Raised when native pipeline output cannot be exported safely."""


def _identity(record: Mapping, field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CandidateExportError(f"{context}.{field} must be a non-empty string")
    return value.strip()


def export_contradiction_candidates(results: Iterable[Mapping]) -> list[dict]:
    output = []
    seen = set()
    for result_index, result in enumerate(results):
        query_id = _identity(result, "query_id", f"results[{result_index}]")
        split = _identity(result, "split", f"results[{result_index}]")
        for pair_index, pair in enumerate(result.get("contradictions", [])):
            first, second = pair.get("paper1"), pair.get("paper2")
            if not isinstance(first, Mapping) or not isinstance(second, Mapping):
                raise CandidateExportError(f"results[{result_index}].contradictions[{pair_index}] lacks papers")
            ids = tuple(sorted((_identity(first, "id", "paper1"), _identity(second, "id", "paper2"))))
            if ids in seen:
                continue
            seen.add(ids)
            papers = {str(first["id"]): dict(first), str(second["id"]): dict(second)}
            verdict = pair.get("verdict")
            candidate_source = {
                "CONTRADICTION": "predicted_contradiction_candidate",
                "AGREEMENT": "agreement_hard_negative_candidate",
                "DIFFERENT SCOPE": "different_scope_hard_negative_candidate",
            }.get(verdict, "terminology_similar_graph_candidate")
            output.append({
                "query_id": query_id, "split": split, "query": result.get("query", ""),
                "paper1": papers[ids[0]], "paper2": papers[ids[1]],
                "candidate_source": candidate_source,
                "shared_concepts": pair.get("shared_concepts"),
                "shared_concept_names": pair.get("shared_concept_names", []),
                "candidate_score": pair.get("candidate_score"),
                "verdict": pair.get("verdict"), "confidence": pair.get("confidence"),
                "concept_jaccard": pair.get("concept_jaccard"),
                "year_gap": pair.get("year_gap"), "flag": pair.get("flag"),
                "llm_analysis": pair.get("llm_analysis"),
                "verdict_valid": pair.get("verdict_valid"),
                "accepted_contradiction": pair.get("accepted_contradiction"),
                "generation_configuration": result.get("generation_configuration"),
            })
    return sorted(output, key=lambda item: (item["split"], item["paper1"]["id"], item["paper2"]["id"]))


def export_claim_support_candidates(
    results: Iterable[Mapping], *, negatives_per_claim: int = 1
) -> list[dict]:
    if isinstance(negatives_per_claim, bool) or not isinstance(negatives_per_claim, int) or negatives_per_claim < 0:
        raise CandidateExportError("negatives_per_claim must be a non-negative integer")
    output = []
    seen = set()
    for result_index, result in enumerate(results):
        query_id = _identity(result, "query_id", f"results[{result_index}]")
        split = _identity(result, "split", f"results[{result_index}]")
        passages = {
            passage["id"]: passage for passage in result.get("passages", [])
            if isinstance(passage, Mapping) and isinstance(passage.get("id"), str)
        }
        papers = {
            str(paper.get("id")): {**paper, "rank": rank}
            for rank, paper in enumerate(result.get("papers", []), start=1)
            if isinstance(paper, Mapping) and paper.get("id")
        }
        all_claims = [
            *((claim, True) for claim in (result.get("claims", []) or [])),
            *((claim, False) for claim in (result.get("unsupported_claims", []) or [])),
        ]
        for claim_index, (claim, grounded) in enumerate(all_claims):
            text = _identity(claim, "text", f"claims[{claim_index}]")
            cited_ids = list(dict.fromkeys(
                claim.get("cited_passage_ids")
                or [item.get("id") for item in claim.get("evidence", []) if isinstance(item, Mapping)]
            ))
            cited_papers = {passages[item]["paper_id"] for item in cited_ids if item in passages}
            selected = [(item, "cited_passage_candidate") for item in cited_ids if item in passages]
            claim_tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))
            negative_candidates = []
            for item, passage in passages.items():
                if item in cited_ids or passage.get("paper_id") in cited_papers:
                    continue
                passage_tokens = set(re.findall(r"[a-z0-9]+", str(passage.get("text", "")).casefold()))
                overlap = len(claim_tokens & passage_tokens) / max(1, len(claim_tokens | passage_tokens))
                paper = papers.get(str(passage.get("paper_id")), {})
                negative_candidates.append((
                    -overlap, int(paper.get("rank", 10**9)), item,
                ))
            negatives = [item for _, _, item in sorted(negative_candidates)[:negatives_per_claim]]
            selected.extend((item, "difficult_negative_candidate") for item in negatives)
            for passage_id, source in selected:
                identity = (query_id, " ".join(text.split()).casefold(), passage_id)
                if identity in seen:
                    continue
                seen.add(identity)
                passage = passages[passage_id]
                paper = papers.get(str(passage.get("paper_id")), {})
                output.append({
                    "query_id": query_id, "split": split, "claim": text,
                    "passage_id": passage_id, "passage_text": passage.get("text", ""),
                    "paper_id": passage.get("paper_id", ""),
                    "candidate_source": source,
                    "retrieval_score": paper.get("score"),
                    "retrieval_source": paper.get("source"),
                    "source_paper_rank": paper.get("rank"),
                    "claim_grounded": grounded,
                    "rejection_reasons": claim.get("rejection_reasons", []),
                    "invalid_passage_ids": claim.get("invalid_passage_ids", []),
                    "generation_configuration": result.get("generation_configuration"),
                })
    return sorted(output, key=lambda item: (item["split"], item["query_id"], item["claim"], item["passage_id"]))


def export_hypothesis_candidates(results: Iterable[Mapping]) -> list[dict]:
    output = []
    seen = set()
    for result_index, result in enumerate(results):
        query_id = _identity(result, "query_id", f"results[{result_index}]")
        split = _identity(result, "split", f"results[{result_index}]")
        items = [
            *(result.get("hypotheses", []) or []),
            *(result.get("rejected_hypotheses", []) or []),
        ]
        for item_index, item in enumerate(items):
            raw = item.get("llm_hypothesis", "")
            parsed = parse_hypothesis_response(raw)
            paper = item.get("paper", {}) if isinstance(item.get("paper"), Mapping) else {}
            text = parsed.get("hypothesis") or item.get("hypothesis") or paper.get("hypothesis")
            if not isinstance(text, str) or not text.strip():
                raise CandidateExportError(f"results[{result_index}].hypothesis[{item_index}] lacks hypothesis text")
            identity = (query_id, " ".join(text.split()).casefold())
            if identity in seen:
                continue
            seen.add(identity)
            output.append({
                "query_id": query_id, "split": split, "hypothesis": text.strip(),
                "evidence": [{
                    "paper_id": paper_id,
                    "shared_concepts": paper.get("shared_concept_names", []),
                    "supporting_evidence": item.get("supporting_evidence") or parsed.get("supporting_evidence", ""),
                } for paper_id in paper.get("supporting_paper_ids", [])],
                "model_feasibility": item.get("feasibility"),
                "accepted": item.get("accepted"),
                "hns": item.get("hns"),
                "candidate_paper_id": paper.get("id"),
                "candidate_score": paper.get("evidence_score"),
                "concept_overlap": paper.get("concept_overlap"),
                "query_support_ratio": paper.get("query_support_ratio"),
                "validation_valid": item.get("validation_valid"),
                "raw_generation": raw,
                "missing_evidence": item.get("missing_evidence") or parsed.get("missing_evidence"),
                "generation_configuration": result.get("generation_configuration"),
            })
    return sorted(output, key=lambda item: (item["split"], item["query_id"], item["hypothesis"]))


EXPORTERS = {
    "contradiction": export_contradiction_candidates,
    "support": export_claim_support_candidates,
    "hypothesis": export_hypothesis_candidates,
}


def _load_results(path: str | Path) -> list[dict]:
    source = Path(path)
    try:
        if source.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateExportError(f"Could not load pipeline results: {error}") from error
    return value if isinstance(value, list) else [value]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(EXPORTERS), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--negatives-per-claim", type=int, default=1)
    arguments = parser.parse_args(argv)
    results = _load_results(arguments.input)
    if arguments.task == "support":
        candidates = export_claim_support_candidates(results, negatives_per_claim=arguments.negatives_per_claim)
    else:
        candidates = EXPORTERS[arguments.task](results)
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for candidate in candidates:
            file.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    print(json.dumps({"task": arguments.task, "candidate_count": len(candidates), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
