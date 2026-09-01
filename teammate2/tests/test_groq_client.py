from types import SimpleNamespace

from src.utils.groq_client import groq_chat_with_retry


def test_groq_wrapper_forwards_reasoning_effort():
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"ok":true}')
            )])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    result = groq_chat_with_retry(
        client, "prompt", model="model-a", reasoning_effort="none"
    )

    assert result == '{"ok":true}'
    assert captured["reasoning_effort"] == "none"
