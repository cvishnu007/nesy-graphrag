"""
src/pipeline/results_logger.py
==============================
Append-only run logger for NeSy-GraphRAG evaluation metrics.

Each call to ``log_result()`` appends one row to a CSV file so that
trends across many queries can be plotted for the evaluation chapter.

CSV columns
-----------
schema_version, timestamp, query, mode, ts, nbr, atd, rdi, hns,
citation_integrity, hallucination_rate, claim_coverage, total_claims,
grounded_claims, total_citations, valid_citations, graph_count,
neural_only_count, total_papers, distinct_years, missing_years,
cross_doc_papers, contradictions_resolved

Usage
-----
    from src.pipeline.results_logger import log_result
    log_result(query, mode, metrics_dict)
"""

import csv
import os
from datetime import datetime, timezone
from typing import Any, Optional

# Default log path (relative to project root)
_DEFAULT_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "evaluation_log_v2.csv"
)

_CSV_COLUMNS = [
    "schema_version",
    "timestamp",
    "query",
    "mode",
    # Top-level scores
    "ts",
    "nbr",
    "atd",
    "rdi",
    "hns",
    # Detail fields
    "citation_integrity",
    "hallucination_rate",
    "claim_coverage",
    "total_claims",
    "grounded_claims",
    "total_citations",
    "valid_citations",
    "graph_count",
    "neural_only_count",
    "total_papers",
    "distinct_years",
    "missing_years",
    "cross_doc_papers",
    "contradictions_resolved",
]


def log_result(
    query: str,
    mode: str,
    metrics: dict[str, Any],
    *,
    log_path: Optional[str] = None,
) -> str:
    """Append one metrics row to the evaluation log CSV.

    Parameters
    ----------
    query   : str  — the research query
    mode    : str  — "review" | "contradict" | "hypothesis" | "baseline"
    metrics : dict — output of ``compute_all_metrics()``
    log_path: str  — optional override for the CSV file path

    Returns
    -------
    str — the absolute path to the log file (for confirmation messages)
    """
    path = os.path.abspath(log_path or _DEFAULT_LOG_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    file_exists = os.path.isfile(path)

    ts_data  = metrics.get("ts", {})
    nbr_data = metrics.get("nbr", {})
    atd_data = metrics.get("atd", {})
    rdi_data = metrics.get("rdi", {})
    hns_data = metrics.get("hns", {})

    row = {
        "schema_version"           : 2,
        "timestamp"               : datetime.now(timezone.utc).isoformat(),
        "query"                   : query,
        "mode"                    : mode,
        "ts"                      : ts_data.get("ts", ""),
        "nbr"                     : nbr_data.get("nbr", ""),
        "atd"                     : atd_data.get("atd", ""),
        "rdi"                     : rdi_data.get("rdi", ""),
        "hns"                     : hns_data.get("hns", ""),
        "citation_integrity"      : ts_data.get("citation_integrity", ""),
        "hallucination_rate"      : ts_data.get("hallucination_rate", ""),
        "claim_coverage"           : ts_data.get("claim_coverage", ""),
        "total_claims"             : ts_data.get("total_claims", ""),
        "grounded_claims"          : ts_data.get("grounded_claims", ""),
        "total_citations"          : ts_data.get("total_citations", ""),
        "valid_citations"          : ts_data.get("valid_citations", ""),
        "graph_count"             : nbr_data.get("graph_count", ""),
        "neural_only_count"       : nbr_data.get("neural_only_count", ""),
        "total_papers"            : nbr_data.get("total", ""),
        "distinct_years"          : str(atd_data.get("distinct_years", [])),
        "missing_years"           : str(atd_data.get("missing_years", [])),
        "cross_doc_papers"        : rdi_data.get("cross_doc_papers", ""),
        "contradictions_resolved" : rdi_data.get("contradictions_resolved", ""),
    }

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return path
