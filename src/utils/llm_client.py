"""Provider-neutral text completion clients for production and evaluation."""

from __future__ import annotations

import time
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import requests

from src.utils.config import (
    GROQ_API_KEY,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_MODEL_FALLBACK,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_CONTEXT_LENGTH,
    OLLAMA_FALLBACK_MODEL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SEC,
    is_configured,
)


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None = None,
    ) -> str:
        """Return one text completion for ``prompt``."""


def _validate_ollama_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("OLLAMA_BASE_URL must be non-empty")
    normalized = base_url.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("OLLAMA_BASE_URL must be an absolute HTTP(S) URL")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError("OLLAMA_BASE_URL contains an invalid port") from error
    return normalized


def _positive_setting(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _safe_url(base_url: str) -> str:
    """Strip user information, query data, and fragments from runtime metadata."""
    parsed = urlsplit(base_url)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        port = ""
    return urlunsplit((parsed.scheme, f"{hostname}{port}", parsed.path.rstrip("/"), "", ""))


class OllamaLLMClient:
    provider = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_sec: int,
        keep_alive: str,
        context_length: int,
        max_retries: int,
        session=None,
        fallback_model: str = "",
    ):
        self.base_url = _validate_ollama_url(base_url)
        self.model = str(model).strip()
        self.fallback_model = str(fallback_model).strip()
        if not self.model:
            raise ValueError("OLLAMA_MODEL must be non-empty")
        self.timeout_sec = _positive_setting("OLLAMA_TIMEOUT_SEC", timeout_sec)
        self.keep_alive = str(keep_alive).strip()
        if not self.keep_alive:
            raise ValueError("OLLAMA_KEEP_ALIVE must be non-empty")
        self.context_length = _positive_setting(
            "OLLAMA_CONTEXT_LENGTH", context_length
        )
        self.max_retries = _positive_setting("LLM_MAX_RETRIES", max_retries)
        self._session = session or requests.Session()
        self._model_verified = False

    def _error_detail(self, response) -> str:
        try:
            payload = response.json()
        except (TypeError, ValueError):
            return ""
        if isinstance(payload, dict) and isinstance(payload.get("error"), str):
            return payload["error"].strip()
        return ""

    def _request(self, method: str, endpoint: str, **kwargs):
        url = f"{self.base_url}{endpoint}"
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = getattr(self._session, method)(
                    url,
                    timeout=self.timeout_sec,
                    **kwargs,
                )
                response.raise_for_status()
                return response
            except (requests.ConnectionError, requests.Timeout) as error:
                last_error = error
            except requests.HTTPError as error:
                response = error.response
                status = getattr(response, "status_code", None)
                detail = self._error_detail(response)
                if status == 404 and "model" in detail.lower():
                    raise RuntimeError(
                        f"Ollama model '{self.model}' is unavailable. "
                        f"Install it with: ollama pull {self.model}"
                    ) from error
                if status is None or status < 500:
                    suffix = f": {detail}" if detail else ""
                    raise RuntimeError(
                        f"Ollama request failed with HTTP {status}{suffix}"
                    ) from error
                last_error = error

            if attempt + 1 < self.max_retries:
                time.sleep(min(30.0, 2.0**attempt))

        if isinstance(last_error, requests.Timeout):
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout_sec} seconds "
                f"at {_safe_url(self.base_url)}"
            ) from last_error
        if isinstance(last_error, requests.ConnectionError):
            raise RuntimeError(
                f"Cannot connect to Ollama at {_safe_url(self.base_url)}. "
                "Start Ollama and check OLLAMA_BASE_URL."
            ) from last_error
        raise RuntimeError(
            f"Ollama server failed after {self.max_retries} attempts at "
            f"{_safe_url(self.base_url)}"
        ) from last_error

    def _verify_model(self) -> None:
        if self._model_verified:
            return
        response = self._request("get", "/api/tags")
        try:
            payload = response.json()
            models = payload["models"]
            installed = {
                value
                for item in models
                if isinstance(item, dict)
                for value in (item.get("name"), item.get("model"))
                if isinstance(value, str)
            }
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Ollama returned an invalid response from /api/tags") from error
        if self.model not in installed and self.fallback_model in installed:
            self.model = self.fallback_model
        if self.model not in installed:
            raise RuntimeError(
                f"Ollama model '{self.model}' is not installed. "
                f"Install it with: ollama pull {self.model}"
            )
        self._model_verified = True

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None = None,
        json_schema: dict | None = None,
    ) -> str:
        self._verify_model()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.context_length,
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema
        response = self._request("post", "/api/chat", json=payload)
        try:
            content = response.json()["message"]["content"]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Ollama returned an invalid response from /api/chat") from error
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an invalid response with empty content")
        return content


class GroqLLMClient:
    provider = "groq"

    def __init__(
        self,
        client,
        *,
        model: str,
        fallback_model: str,
        max_retries: int,
    ):
        self._client = client
        self.model = model
        self.fallback_model = fallback_model
        self.max_retries = max_retries

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None = None,
    ) -> str:
        from src.utils.groq_client import groq_chat_with_retry

        return groq_chat_with_retry(
            self._client,
            prompt,
            model=self.model,
            fallback_model=self.fallback_model,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=self.max_retries,
            reasoning_effort=reasoning_effort,
        )


def create_llm_client() -> LLMClient:
    """Construct only the configured provider implementation."""
    if LLM_PROVIDER == "ollama":
        return OllamaLLMClient(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            fallback_model=OLLAMA_FALLBACK_MODEL,
            timeout_sec=OLLAMA_TIMEOUT_SEC,
            keep_alive=OLLAMA_KEEP_ALIVE,
            context_length=OLLAMA_CONTEXT_LENGTH,
            max_retries=LLM_MAX_RETRIES,
        )
    if LLM_PROVIDER == "groq":
        if not is_configured(GROQ_API_KEY):
            raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        from groq import Groq

        return GroqLLMClient(
            Groq(api_key=GROQ_API_KEY),
            model=LLM_MODEL,
            fallback_model=LLM_MODEL_FALLBACK,
            max_retries=_positive_setting("LLM_MAX_RETRIES", LLM_MAX_RETRIES),
        )
    raise ValueError("LLM_PROVIDER must be 'ollama' or 'groq'")


def llm_runtime_metadata() -> dict:
    """Describe the selected provider without constructing it or exposing secrets."""
    if LLM_PROVIDER == "ollama":
        _validate_ollama_url(OLLAMA_BASE_URL)
        return {
            "provider": "ollama",
            "model": OLLAMA_MODEL,
            "base_url": _safe_url(OLLAMA_BASE_URL),
            "timeout_sec": OLLAMA_TIMEOUT_SEC,
            "max_retries": LLM_MAX_RETRIES,
            "context_length": OLLAMA_CONTEXT_LENGTH,
            "keep_alive": OLLAMA_KEEP_ALIVE,
        }
    if LLM_PROVIDER == "groq":
        return {
            "provider": "groq",
            "model": LLM_MODEL,
            "base_url": "https://api.groq.com",
            "timeout_sec": None,
            "max_retries": LLM_MAX_RETRIES,
        }
    raise ValueError("LLM_PROVIDER must be 'ollama' or 'groq'")
