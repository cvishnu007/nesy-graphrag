"""Create a read-only manifest for frozen reasoning-evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from src.utils import config


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_fingerprint(path: str | Path) -> dict:
    root = Path(path)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        relative = item.relative_to(root).as_posix()
        item_hash = sha256_file(item)
        size = item.stat().st_size
        total_bytes += size
        digest.update(f"{relative}\0{size}\0{item_hash}\n".encode("utf-8"))
    return {
        "path": str(root), "file_count": len(files), "total_bytes": total_bytes,
        "sha256_manifest": digest.hexdigest(),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def inspect_corpus(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_file():
        return {"status": "unavailable", "path": str(source)}
    try:
        records = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "invalid", "path": str(source), "error_type": type(error).__name__}
    if not isinstance(records, list):
        return {"status": "invalid", "path": str(source), "error_type": "not_a_list"}
    identifiers = [str(record.get("id") or "") for record in records if isinstance(record, dict)]
    years = [record.get("year") for record in records if isinstance(record, dict) and isinstance(record.get("year"), int)]
    return {
        "status": "verified",
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
        "paper_count": len(records),
        "nonempty_id_count": sum(bool(value) for value in identifiers),
        "unique_id_count": len(set(identifiers)),
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
    }


def inspect_chroma(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_dir():
        return {"status": "unavailable", "path": str(source)}
    fingerprint = directory_fingerprint(source)
    try:
        client = chromadb.PersistentClient(path=str(source))
        collections = [
            {"name": collection.name, "count": collection.count(), "metadata": collection.metadata or {}}
            for collection in sorted(client.list_collections(), key=lambda item: item.name)
        ]
    except Exception as error:
        return {**fingerprint, "status": "invalid", "error_type": type(error).__name__}
    return {**fingerprint, "status": "verified", "collections": collections}


def inspect_neo4j() -> dict:
    if not all((
        config.is_configured(config.NEO4J_URI, "YOUR_INSTANCE"),
        config.is_configured(config.NEO4J_USERNAME),
        config.is_configured(config.NEO4J_PASSWORD),
    )):
        return {"status": "unavailable", "reason": "Neo4j credentials are incomplete"}
    try:
        from src.storage.neo4j_store import get_driver

        driver = get_driver()
        try:
            with driver.session() as session:
                counts = {}
                for label in ("Paper", "Author", "Concept"):
                    counts[label.lower()] = session.run(
                        f"MATCH (n:{label}) RETURN count(n) AS c"
                    ).single()["c"]
                counts["cites"] = session.run(
                    "MATCH ()-[r:CITES]->() RETURN count(r) AS c"
                ).single()["c"]
        finally:
            driver.close()
        return {"status": "verified", "uri": config.NEO4J_URI, "counts": counts}
    except Exception as error:
        return {
            "status": "unavailable", "uri": config.NEO4J_URI,
            "reason": "Neo4j read-only connectivity/count check failed",
            "error_type": type(error).__name__,
        }


def inspect_query_benchmark(path: str | Path) -> dict:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
        queries = data["queries"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        return {"status": "invalid", "path": str(source), "error_type": type(error).__name__}
    splits = {}
    for query in queries:
        split = query.get("split")
        splits[split] = splits.get(split, 0) + 1
    return {
        "status": "verified", "path": str(source), "sha256": sha256_file(source),
        "benchmark_version": data.get("benchmark_version"),
        "benchmark_status": data.get("status"), "query_count": len(queries),
        "splits": dict(sorted(splits.items())), "corpus_declaration": data.get("corpus"),
    }


def create_artifact_manifest(
    *, corpus_path: str | Path = "data/s2_clean.json",
    chroma_path: str | Path = "data/chromadb",
    query_benchmark_path: str | Path = "evaluation/benchmarks/retrieval_queries_judged.json",
) -> dict:
    corpus = inspect_corpus(corpus_path)
    chroma = inspect_chroma(chroma_path)
    neo4j = inspect_neo4j()
    query_benchmark = inspect_query_benchmark(query_benchmark_path)
    s2_collection = next(
        (item for item in chroma.get("collections", []) if item["name"] == config.CHROMA_COLLECTION),
        None,
    )
    consistency = {
        "corpus_matches_configured_chroma": bool(
            corpus.get("paper_count") is not None and s2_collection
            and corpus["paper_count"] == s2_collection["count"]
        ),
        "corpus_matches_query_declaration": bool(
            corpus.get("paper_count") is not None
            and query_benchmark.get("corpus_declaration", {}).get("paper_count") == corpus["paper_count"]
        ),
    }
    components = (corpus, chroma, neo4j, query_benchmark)
    overall = "verified" if all(item.get("status") == "verified" for item in components) and all(consistency.values()) else "partial"
    try:
        spacy_model_version = importlib.metadata.version("en-core-web-sm")
    except importlib.metadata.PackageNotFoundError:
        spacy_model_version = None
    prompts_path = Path("src/pipeline/prompts.py")
    return {
        "manifest_version": "1.0",
        "verification_status": overall,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "read_only_verification": True,
        "corpus": corpus,
        "chroma": chroma,
        "neo4j": neo4j,
        "query_benchmark": query_benchmark,
        "consistency": consistency,
        "models": {
            "ner": {"model": "en_core_web_sm", "version": spacy_model_version},
            "embedding": config.EMBEDDING_MODEL,
            "llm_primary": config.LLM_MODEL,
            "llm_fallback": config.LLM_MODEL_FALLBACK,
            "semantic_support": config.SEMANTIC_SUPPORT_MODEL,
        },
        "configuration": {
            "data_source": config.DATA_SOURCE,
            "chroma_collection": config.CHROMA_COLLECTION,
            "contradiction_min_shared_concepts": config.CONTRADICTION_MIN_SHARED_CONCEPTS,
            "contradiction_min_concept_jaccard": config.CONTRADICTION_MIN_CONCEPT_JACCARD,
            "contradiction_candidate_pool": config.CONTRADICTION_CANDIDATE_POOL,
            "contradiction_min_confidence": config.CONTRADICTION_MIN_CONFIDENCE,
            "hypothesis_min_shared_concepts": config.HYPOTHESIS_MIN_SHARED_CONCEPTS,
            "hypothesis_min_query_support": config.HYPOTHESIS_MIN_QUERY_SUPPORT,
            "hypothesis_candidate_pool": config.HYPOTHESIS_CANDIDATE_POOL,
            "semantic_support_min_confidence": config.SEMANTIC_SUPPORT_MIN_CONFIDENCE,
        },
        "prompts": {
            "path": str(prompts_path),
            "sha256": sha256_file(prompts_path) if prompts_path.is_file() else None,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/s2_clean.json")
    parser.add_argument("--chroma", default="data/chromadb")
    parser.add_argument("--query-benchmark", default="evaluation/benchmarks/retrieval_queries_judged.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    output = Path(arguments.output)
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(f"Refusing to overwrite artifact manifest: {output}")
    manifest = create_artifact_manifest(
        corpus_path=arguments.corpus, chroma_path=arguments.chroma,
        query_benchmark_path=arguments.query_benchmark,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verification_status": manifest["verification_status"], "output": str(output)}, indent=2))
    return 0 if manifest["verification_status"] in {"verified", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
