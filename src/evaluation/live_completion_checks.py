"""Run and record non-destructive final integration checks."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable


CHECKS = {
    "chroma": {"mutates_state": False, "external": False},
    "neo4j": {"mutates_state": False, "external": True},
    "local_nli": {"mutates_state": False, "external": False},
    "groq": {"mutates_state": False, "external": True},
    "semantic_review": {"mutates_state": False, "external": True},
    "contradiction_path": {"mutates_state": False, "external": True},
    "hypothesis_path": {"mutates_state": False, "external": True},
}

EVALUATION_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


def run_check(name: str, operation: Callable[[], object]) -> dict:
    """Execute a check and preserve failures as explicit structured evidence."""
    started = perf_counter()
    try:
        details = operation()
        return {
            "check": name,
            "status": "passed",
            "mutates_state": False,
            "runtime_seconds": perf_counter() - started,
            "details": details,
        }
    except Exception as error:
        return {
            "check": name,
            "status": "failed",
            "mutates_state": False,
            "runtime_seconds": perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _chroma_check() -> dict:
    import chromadb
    from src.utils.config import CHROMA_COLLECTION, CHROMA_DIR

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(CHROMA_COLLECTION)
    count = collection.count()
    if count <= 0:
        raise RuntimeError("configured Chroma collection is empty")
    sample = collection.peek(limit=1)
    return {"path": CHROMA_DIR, "collection": CHROMA_COLLECTION, "count": count,
            "read_sample_count": len(sample.get("ids", []))}


def _neo4j_check() -> dict:
    from src.storage.neo4j_store import get_driver

    driver = get_driver()
    try:
        with driver.session() as session:
            paper_count = session.run("MATCH (p:Paper) RETURN count(p) AS count").single()["count"]
        return {"paper_count": paper_count, "query": "read-only count"}
    finally:
        driver.close()


def _local_nli_check() -> dict:
    from src.evaluation.semantic_support import build_local_nli_provider
    from src.utils.config import SEMANTIC_SUPPORT_MODEL

    model = (
        EVALUATION_NLI_MODEL
        if SEMANTIC_SUPPORT_MODEL == "unconfigured"
        else SEMANTIC_SUPPORT_MODEL
    )
    provider = build_local_nli_provider(model)
    prediction = provider(
        "Graph neural networks operate on graphs.",
        {"id": "live-P1", "text": "Graph neural networks process graph-structured data."},
    )
    if not prediction.get("valid"):
        raise RuntimeError("local NLI returned an invalid decision")
    return {
        "model": model,
        "production_configuration_overridden": SEMANTIC_SUPPORT_MODEL == "unconfigured",
        "prediction": prediction,
    }


def _clients():
    from groq import Groq
    from src.storage.neo4j_store import get_driver
    from src.utils.config import GROQ_API_KEY

    return Groq(api_key=GROQ_API_KEY), get_driver()


def _groq_check() -> dict:
    from groq import Groq
    from src.utils.config import GROQ_API_KEY

    models = Groq(api_key=GROQ_API_KEY).models.list().data
    return {"model_count": len(models), "model_ids": sorted(model.id for model in models)}


def _pipeline_check(mode: str) -> dict:
    from src.pipeline.contradiction import llm_contradict
    from src.pipeline.hypothesis import llm_hypothesis
    from src.pipeline.review import llm_review
    from src.utils.config import SEMANTIC_SUPPORT_MODEL

    client, driver = _clients()
    try:
        query = "graph neural network methods"
        if mode == "review":
            support_model = EVALUATION_NLI_MODEL if SEMANTIC_SUPPORT_MODEL == "unconfigured" else SEMANTIC_SUPPORT_MODEL
            result = llm_review(client, driver, query, top_k=2, support_model=support_model)
            return {"paper_count": len(result["papers"]), "claim_count": len(result["claims"]),
                    "semantic_support": result["semantic_support"]}
        if mode == "contradiction":
            result = llm_contradict(client, driver, query, top_k=1)
            return {"pair_count": len(result["contradictions"])}
        result = llm_hypothesis(client, driver, query, top_k=1)
        return {"hypothesis_count": len(result["hypotheses"])}
    finally:
        driver.close()


def execute_checks(*, external_authorized: bool) -> list[dict]:
    operations = {
        "chroma": _chroma_check,
        "neo4j": _neo4j_check,
        "local_nli": _local_nli_check,
        "groq": _groq_check,
        "semantic_review": lambda: _pipeline_check("review"),
        "contradiction_path": lambda: _pipeline_check("contradiction"),
        "hypothesis_path": lambda: _pipeline_check("hypothesis"),
    }
    results = []
    for name, spec in CHECKS.items():
        if spec["external"] and not external_authorized:
            results.append({
                "check": name,
                "status": "not_run",
                "mutates_state": False,
                "reason": "external access was not authorized for this run",
            })
        else:
            results.append(run_check(name, operations[name]))
    return results


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--external-authorized", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = execute_checks(external_authorized=args.external_authorized)
    for row in results:
        (output / f"{row['check']}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures = [row for row in results if row["status"] == "failed"]
    not_run = [row for row in results if row["status"] == "not_run"]
    summary = {
        "read_only": True,
        "passed": sum(row["status"] == "passed" for row in results),
        "failed": len(failures),
        "not_run": len(not_run),
        "all_required_passed": not failures and not not_run,
        "checks": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "failures.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures), encoding="utf-8")
    command = ".\\venv\\Scripts\\python.exe -m src.evaluation.live_completion_checks --output-dir " + args.output_dir
    if args.external_authorized:
        command += " --external-authorized"
    metadata = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "external_authorized": args.external_authorized,
        "credentials_saved": False,
        "read_only": True,
        "reproduction_command": command,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "reproduce.ps1").write_text(command + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("passed", "failed", "not_run", "all_required_passed")}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
