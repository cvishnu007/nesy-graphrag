from src.evaluation.live_completion_checks import CHECKS, run_check


def test_live_check_failure_is_saved_not_hidden():
    result = run_check(
        "neo4j", lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert "offline" in result["error"]


def test_live_checks_are_declared_non_destructive():
    assert CHECKS
    assert all(spec["mutates_state"] is False for spec in CHECKS.values())
    assert CHECKS["chroma"]["mutates_state"] is False
    assert CHECKS["neo4j"]["mutates_state"] is False
