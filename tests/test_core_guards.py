import unittest
from unittest.mock import patch

from src.pipeline.metrics import compute_all_metrics
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


class CoreGuardTests(unittest.TestCase):
    def test_validator_blocks_fabricated_paper_id(self):
        driver = FakeDriver(records=[{"id": "real", "title": "Real paper"}])

        verified = validate_citations(driver, ["real", "fabricated"])

        self.assertEqual(verified, {"real": "Real paper"})
        self.assertNotIn("fabricated", verified)

    def test_all_metrics_handle_empty_result(self):
        scores = compute_all_metrics({"papers": [], "answer": "", "verified": {}})

        self.assertEqual(scores["ts"]["ts"], 0.0)
        self.assertEqual(scores["nbr"]["nbr"], 0.0)
        self.assertEqual(scores["atd"]["atd"], 0.0)
        self.assertEqual(scores["rdi"]["rdi"], 0.0)
        self.assertEqual(scores["hns"]["hns"], 0.0)

    def test_missing_neo4j_credentials_fail_before_driver_creation(self):
        with (
            patch.object(neo4j_store, "NEO4J_URI", None),
            patch.object(neo4j_store, "NEO4J_USERNAME", None),
            patch.object(neo4j_store, "NEO4J_PASSWORD", None),
            patch.object(neo4j_store.GraphDatabase, "driver") as driver_factory,
        ):
            with self.assertRaisesRegex(RuntimeError, "Missing Neo4j credentials"):
                neo4j_store.get_driver()

        driver_factory.assert_not_called()

    def test_connectivity_failure_is_clear_and_closes_driver(self):
        driver = FakeDriver(connectivity_error=OSError("connection refused"))
        with (
            patch.object(neo4j_store, "NEO4J_URI", "neo4j://127.0.0.1:7687"),
            patch.object(neo4j_store, "NEO4J_USERNAME", "neo4j"),
            patch.object(neo4j_store, "NEO4J_PASSWORD", "secret"),
            patch.object(neo4j_store.GraphDatabase, "driver", return_value=driver),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Could not connect to Neo4j at neo4j://127.0.0.1:7687",
            ) as context:
                neo4j_store.get_driver()

        self.assertTrue(driver.closed)
        self.assertNotIn("secret", str(context.exception))


if __name__ == "__main__":
    unittest.main()
