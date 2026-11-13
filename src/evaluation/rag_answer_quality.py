"""Score matched-RAG answers for semantic support, not only citation validity.

``matched_rag`` checks that cited passage IDs exist. This adds an NLI check of
whether the cited passages entail each accepted claim (the same local NLI
provider used by review mode), and reports per arm:

* claims per answer and abstention rate (no accepted claim);
* structural citation precision (valid IDs / cited IDs);
* semantic support rate: claims judged SUPPORTED or PARTIALLY_SUPPORTED;
* contradicted rate: claims the cited evidence contradicts.

Paired by query, so arms are compared on identical questions. NLI support is an
automated proxy that needs a human-checked sample before publication.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


def score_reviews(rows, provider, *, min_confidence=0.5):
    from src.evaluation.semantic_support import verify_claim_support

    per_arm = defaultdict(lambda: defaultdict(list))
    per_query = defaultdict(dict)
    for row in rows:
        claims = row["provenance"]["claims"]
        passages = {p["id"]: p for p in row["passages"]}
        labels = []
        for claim in claims:
            evidence = [passages[i] for i in claim.get("cited_passage_ids", []) if i in passages]
            decision = verify_claim_support(claim["text"], evidence, provider=provider,
                                            model=NLI_MODEL, min_confidence=min_confidence)
            labels.append(decision["support_label"] if decision["valid"] else "UNSUPPORTED")
        stats = row["provenance"]["stats"]
        supported = sum(label in {"SUPPORTED", "PARTIALLY_SUPPORTED"} for label in labels)
        arm = per_arm[row["method"]]
        arm["claims"].append(len(claims))
        arm["abstained"].append(int(not claims))
        arm["citation_precision"].append(stats["citation_precision"] if stats["total_citations"] else None)
        arm["supported"].append(supported)
        arm["contradicted"].append(sum(label == "CONTRADICTED" for label in labels))
        per_query[row["query_id"]][row["method"]] = supported / len(claims) if claims else 0.0
    summary = {}
    for method, values in per_arm.items():
        total_claims = sum(values["claims"])
        precisions = [v for v in values["citation_precision"] if v is not None]
        summary[method] = {
            "queries": len(values["claims"]),
            "mean_claims_per_answer": round(total_claims / len(values["claims"]), 3),
            "abstention_rate": round(sum(values["abstained"]) / len(values["abstained"]), 4),
            "structural_citation_precision": round(sum(precisions) / len(precisions), 4) if precisions else None,
            "semantic_support_rate": round(sum(values["supported"]) / total_claims, 4) if total_claims else None,
            "contradicted_rate": round(sum(values["contradicted"]) / total_claims, 4) if total_claims else None,
            "total_claims": total_claims,
        }
    return summary, dict(per_query)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews", required=True, help="matched_rag reviews.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args(argv)
    from src.evaluation.semantic_support import build_local_nli_provider
    from src.evaluation.significance import paired_significance

    rows = [json.loads(line) for line in Path(args.reviews).read_text(encoding="utf-8").splitlines() if line.strip()]
    summary, per_query = score_reviews(rows, build_local_nli_provider(NLI_MODEL), min_confidence=args.min_confidence)
    methods = sorted(summary)
    test = None
    if len(methods) == 2:
        shared = sorted(q for q, v in per_query.items() if all(m in v for m in methods))
        deltas = [per_query[q][methods[0]] - per_query[q][methods[1]] for q in shared]
        if 0 < len(deltas) <= 20:
            test = {"comparison": f"{methods[0]} - {methods[1]} per-query support rate", **paired_significance(deltas)}
    report = {"nli_model": NLI_MODEL, "min_confidence": args.min_confidence, "arms": summary,
              "paired_test": test, "per_query_support_rate": per_query,
              "note": "Automated NLI support; validate on a human-labelled sample before publication."}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"arms": summary, "paired_test": test}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
