import csv

import pytest

from src.pipeline.metrics import compute_all_metrics, compute_atd
from src.pipeline.results_logger import log_result
from src.pipeline.validator import validate_citations
from src.storage import neo4j_store


class FakeSession:
    def __init__(self, records):
        self.records = records

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def run(self, query, **parameters):
        return self.records


class FakeDriver:
    def __init__(self, records=(), connectivity_error=None):
        self.records = records
        self.connectivity_error = connectivity_error
        self.closed = False

    def session(self):
        return FakeSession(self.records)

    def verify_connectivity(self):
        if self.connectivity_error:
            raise self.connectivity_error

    def close(self):
        self.closed = True


def test_validator_blocks_fabricated_paper_id(capsys):
    driver = FakeDriver(records=[{"id": "real", "title": "Real paper"}])

    verified = validate_citations(driver, ["real", "fabricated"])

    assert verified == {"real": "Real paper"}
    assert "fabricated" not in verified
    assert "Blocked" in capsys.readouterr().out


def test_all_metrics_handle_empty_result(capsys):
    scores = compute_all_metrics({"papers": [], "answer": "", "verified": {}})

    assert scores["ts"]["ts"] == 0.0
    assert scores["nbr"]["nbr"] == 0.0
    assert scores["atd"]["atd"] == 0.0
    assert scores["rdi"]["rdi"] == 0.0
    assert scores["hns"]["hns"] == 0.0
    capsys.readouterr()


def test_atd_uses_configured_s2_year_range():
    result = compute_atd([{"year": 2020}, {"year": 2025}])

    assert result["span_size"] == 6
    assert result["atd"] == pytest.approx(2 / 6, abs=0.0001)


def test_missing_neo4j_credentials_fail_before_driver_creation(monkeypatch):
    driver_called = False

    def unexpected_driver(*args, **kwargs):
        nonlocal driver_called
        driver_called = True

    monkeypatch.setattr(neo4j_store, "NEO4J_URI", None)
    monkeypatch.setattr(neo4j_store, "NEO4J_USERNAME", None)
    monkeypatch.setattr(neo4j_store, "NEO4J_PASSWORD", None)
    monkeypatch.setattr(neo4j_store.GraphDatabase, "driver", unexpected_driver)

    with pytest.raises(RuntimeError, match="Missing Neo4j credentials"):
        neo4j_store.get_driver()

    assert driver_called is False


def test_connectivity_failure_is_clear_and_closes_driver(monkeypatch):
    driver = FakeDriver(connectivity_error=OSError("connection refused"))
    monkeypatch.setattr(neo4j_store, "NEO4J_URI", "neo4j://127.0.0.1:7687")
    monkeypatch.setattr(neo4j_store, "NEO4J_USERNAME", "neo4j")
    monkeypatch.setattr(neo4j_store, "NEO4J_PASSWORD", "secret")
    monkeypatch.setattr(
        neo4j_store.GraphDatabase,
        "driver",
        lambda *args, **kwargs: driver,
    )

    with pytest.raises(
        RuntimeError,
        match="Could not connect to Neo4j at neo4j://127.0.0.1:7687",
    ) as error:
        neo4j_store.get_driver()

    assert driver.closed is True
    assert "secret" not in str(error.value)


def test_graph_rebuild_requires_explicit_reset_opt_in(monkeypatch):
    monkeypatch.setattr(neo4j_store, "NEO4J_ALLOW_RESET", False)

    with pytest.raises(RuntimeError, match="NEO4J_ALLOW_RESET=true"):
        neo4j_store.insert_papers(object(), [])


def test_results_logger_writes_provenance_schema(tmp_path):
    log_path = tmp_path / "evaluation.csv"
    metrics = {
        "ts": {
            "ts": 1.0,
            "citation_integrity": 1.0,
            "hallucination_rate": 0.0,
            "claim_coverage": 1.0,
            "total_claims": 2,
            "grounded_claims": 2,
            "total_citations": 3,
            "valid_citations": 3,
        }
    }

    log_result("query", "review", metrics, log_path=str(log_path))

    with log_path.open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["schema_version"] == "2"
    assert row["claim_coverage"] == "1.0"
    assert row["valid_citations"] == "3"
