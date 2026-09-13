import builtins

import pytest
import requests

from src.utils import llm_client


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {self.status_code}",
                response=self,
            )


class FakeSession:
    def __init__(self, *, gets=(), posts=()):
        self.gets = iter(gets)
        self.posts = iter(posts)
        self.get_calls = []
        self.post_calls = []

    def get(self, url, *, timeout):
        self.get_calls.append({"url": url, "timeout": timeout})
        result = next(self.gets)
        if isinstance(result, Exception):
            raise result
        return result

    def post(self, url, *, json, timeout):
        self.post_calls.append({"url": url, "json": json, "timeout": timeout})
        result = next(self.posts)
        if isinstance(result, Exception):
            raise result
        return result


def configure_ollama(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm_client, "GROQ_API_KEY", None)
    monkeypatch.setattr(llm_client, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr(llm_client, "OLLAMA_MODEL", "qwen2.5:3b-instruct")
    monkeypatch.setattr(llm_client, "OLLAMA_TIMEOUT_SEC", 300)
    monkeypatch.setattr(llm_client, "OLLAMA_KEEP_ALIVE", "10m")
    monkeypatch.setattr(llm_client, "OLLAMA_CONTEXT_LENGTH", 8192)
    monkeypatch.setattr(llm_client, "LLM_MAX_RETRIES", 3)


def test_factory_creates_ollama_without_groq_key_or_import(monkeypatch):
    configure_ollama(monkeypatch)
    imported = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "groq" or name.startswith("groq."):
            imported.append(name)
            raise AssertionError("Ollama selection imported Groq")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    client = llm_client.create_llm_client()

    assert isinstance(client, llm_client.OllamaLLMClient)
    assert imported == []


def test_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        llm_client.create_llm_client()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OLLAMA_BASE_URL", ""),
        ("OLLAMA_BASE_URL", "localhost:11434"),
        ("OLLAMA_MODEL", ""),
        ("OLLAMA_KEEP_ALIVE", ""),
        ("OLLAMA_TIMEOUT_SEC", 0),
        ("OLLAMA_CONTEXT_LENGTH", 0),
        ("LLM_MAX_RETRIES", 0),
    ],
)
def test_factory_rejects_malformed_ollama_settings(monkeypatch, name, value):
    configure_ollama(monkeypatch)
    monkeypatch.setattr(llm_client, name, value)

    with pytest.raises(ValueError, match=name):
        llm_client.create_llm_client()


def test_ollama_completion_verifies_model_once_and_maps_payload():
    session = FakeSession(
        gets=[FakeResponse({"models": [{"name": "qwen2.5:3b-instruct"}]})],
        posts=[
            FakeResponse({"message": {"content": "first answer"}}),
            FakeResponse({"message": {"content": "second answer"}}),
        ],
    )
    client = llm_client.OllamaLLMClient(
        base_url="http://127.0.0.1:11434/",
        model="qwen2.5:3b-instruct",
        timeout_sec=300,
        keep_alive="10m",
        context_length=8192,
        max_retries=3,
        session=session,
    )

    assert client.complete("prompt", max_tokens=300, temperature=0.0) == "first answer"
    assert client.complete("again", max_tokens=100, temperature=0.2) == "second answer"

    assert session.get_calls == [
        {"url": "http://127.0.0.1:11434/api/tags", "timeout": 300}
    ]
    assert session.post_calls[0] == {
        "url": "http://127.0.0.1:11434/api/chat",
        "timeout": 300,
        "json": {
            "model": "qwen2.5:3b-instruct",
            "messages": [{"role": "user", "content": "prompt"}],
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.0,
                "num_predict": 300,
                "num_ctx": 8192,
            },
        },
    }


def test_missing_ollama_model_is_actionable_and_not_retried():
    session = FakeSession(gets=[FakeResponse({"models": [{"name": "other:latest"}]})])
    client = llm_client.OllamaLLMClient(
        base_url="http://127.0.0.1:11434",
        model="qwen2.5:3b-instruct",
        timeout_sec=10,
        keep_alive="10m",
        context_length=4096,
        max_retries=3,
        session=session,
    )

    with pytest.raises(RuntimeError, match=r"ollama pull qwen2\.5:3b-instruct"):
        client.complete("prompt", max_tokens=10, temperature=0.0)

    assert len(session.get_calls) == 1
    assert session.post_calls == []


def test_retryable_ollama_failures_stop_at_configured_attempt_bound(monkeypatch):
    session = FakeSession(
        gets=[FakeResponse({"models": [{"model": "qwen2.5:3b-instruct"}]})],
        posts=[
            requests.ConnectionError("refused"),
            FakeResponse({"error": "busy"}, status_code=503),
            requests.Timeout("slow"),
        ],
    )
    client = llm_client.OllamaLLMClient(
        base_url="http://127.0.0.1:11434",
        model="qwen2.5:3b-instruct",
        timeout_sec=1,
        keep_alive="10m",
        context_length=4096,
        max_retries=3,
        session=session,
    )
    sleeps = []
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="timed out"):
        client.complete("prompt", max_tokens=10, temperature=0.0)

    assert len(session.post_calls) == 3
    assert sleeps == [1.0, 2.0]


def test_ollama_client_error_is_immediate_and_missing_model_payload_is_actionable(
    monkeypatch,
):
    session = FakeSession(
        gets=[FakeResponse({"models": [{"name": "qwen2.5:3b-instruct"}]})],
        posts=[
            FakeResponse({"error": "model not found"}, status_code=404),
            AssertionError("must not retry a client error"),
        ],
    )
    client = llm_client.OllamaLLMClient(
        base_url="http://127.0.0.1:11434",
        model="qwen2.5:3b-instruct",
        timeout_sec=10,
        keep_alive="10m",
        context_length=4096,
        max_retries=3,
        session=session,
    )
    monkeypatch.setattr(
        llm_client.time,
        "sleep",
        lambda _: pytest.fail("client errors must not sleep"),
    )

    with pytest.raises(RuntimeError, match=r"ollama pull qwen2\.5:3b-instruct"):
        client.complete("prompt", max_tokens=10, temperature=0.0)

    assert len(session.post_calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": {}},
        {"message": {"content": "   "}},
        {"message": {"content": 42}},
    ],
)
def test_successful_ollama_response_requires_nonempty_text(payload):
    session = FakeSession(
        gets=[FakeResponse({"models": [{"name": "qwen2.5:3b-instruct"}]})],
        posts=[FakeResponse(payload)],
    )
    client = llm_client.OllamaLLMClient(
        base_url="http://127.0.0.1:11434",
        model="qwen2.5:3b-instruct",
        timeout_sec=10,
        keep_alive="10m",
        context_length=4096,
        max_retries=3,
        session=session,
    )

    with pytest.raises(RuntimeError, match="invalid response"):
        client.complete("prompt", max_tokens=10, temperature=0.0)

    assert len(session.post_calls) == 1


def test_runtime_metadata_removes_credentials_query_and_fragment(monkeypatch):
    configure_ollama(monkeypatch)
    monkeypatch.setattr(
        llm_client,
        "OLLAMA_BASE_URL",
        "http://user:secret@localhost:11434/prefix?token=private#fragment",
    )

    metadata = llm_client.llm_runtime_metadata()

    assert metadata == {
        "provider": "ollama",
        "model": "qwen2.5:3b-instruct",
        "base_url": "http://localhost:11434/prefix",
        "timeout_sec": 300,
        "max_retries": 3,
        "context_length": 8192,
        "keep_alive": "10m",
    }
    assert "secret" not in str(metadata)
    assert "private" not in str(metadata)


def test_ollama_uses_explicit_installed_fallback_when_primary_is_missing(monkeypatch):
    configure_ollama(monkeypatch)
    monkeypatch.setattr(llm_client, 'OLLAMA_FALLBACK_MODEL', 'other:latest', raising=False)
    session=FakeSession(gets=[FakeResponse({'models':[{'name':'other:latest'}]})], posts=[FakeResponse({'message':{'content':'fallback answer'}})])
    monkeypatch.setattr(llm_client.requests,'Session',lambda:session)
    client=llm_client.create_llm_client()
    assert client.complete('q',max_tokens=10,temperature=0.0)=='fallback answer'
    assert session.post_calls[0]['json']['model']=='other:latest'
    assert client.model=='other:latest'


def test_groq_adapter_preserves_explicit_experiment_parameters(monkeypatch):
    from src.utils import groq_client
    calls=[]
    def complete(client,prompt,**options):
        calls.append(options); return 'hosted answer'
    monkeypatch.setattr(groq_client,'groq_chat_with_retry',complete)
    client=llm_client.GroqLLMClient(object(),model='primary',fallback_model='fallback',max_retries=4)
    assert client.complete('q',max_tokens=17,temperature=0.0,reasoning_effort='none') == 'hosted answer'
    assert calls == [{'model':'primary','fallback_model':'fallback','max_tokens':17,'temperature':0.0,'max_retries':4,'reasoning_effort':'none'}]
