from src.evaluation import live_completion_checks
from src.evaluation.live_completion_checks import CHECKS, run_check


class FakeLLMClient:
    provider = "ollama"
    model = "qwen2.5:3b-instruct"

    def __init__(self):
        self.calls = []

    def complete(self, prompt, **options):
        self.calls.append({"prompt": prompt, **options})
        return "ready"


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


def test_selected_llm_check_reports_provider_model_and_uses_common_contract():
    client = FakeLLMClient()

    result = live_completion_checks._llm_check(
        client_factory=lambda: client,
        metadata_provider=lambda: {
            "provider": "ollama",
            "model": "qwen2.5:3b-instruct",
            "base_url": "http://127.0.0.1:11434",
        },
    )

    assert result["provider"] == "ollama"
    assert result["model"] == "qwen2.5:3b-instruct"
    assert result["completion"] == "ready"
    assert client.calls == [
        {"prompt": "Reply with exactly: ready", "max_tokens": 16, "temperature": 0.0}
    ]
    assert "llm" in CHECKS
    assert "groq" not in CHECKS


def test_pipeline_smoke_rejects_empty_or_invalid_outputs():
    import pytest
    from src.evaluation.live_completion_checks import pipeline_evidence
    with pytest.raises(RuntimeError, match='valid verdict'):
        pipeline_evidence('contradiction', {'contradictions': [{'verdict_valid':False}]})
    with pytest.raises(RuntimeError, match='evaluated claims'):
        pipeline_evidence('review', {'papers':[{}], 'claims':[], 'semantic_support':{'stats':{'evaluated_claims':0}}})
    details=pipeline_evidence('hypothesis', {'hypotheses':[], 'rejected_hypotheses':[{'validation_valid':True}]})
    assert details['hypothesis_count']==0 and details['valid_generation_count']==1
