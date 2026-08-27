"""
src/utils/groq_client.py
========================
Thin retry/backoff wrapper around ``groq.Groq`` chat completions.

Reuses the same exponential-backoff strategy already proven in
``SemanticScholarClient`` (semantic_scholar_fetcher.py) so every LLM
call in the pipeline gets resilient error handling for free.

Usage
-----
    from src.utils.groq_client import groq_chat_with_retry

    answer = groq_chat_with_retry(
        groq_client, prompt,
        model=LLM_MODEL,
        fallback_model=LLM_MODEL_FALLBACK,
    )
"""

import time
from typing import Optional


# Defaults (can be overridden per-call)
_DEFAULT_MAX_RETRIES = 4
_DEFAULT_MAX_TOKENS  = 1024
_DEFAULT_TEMPERATURE = 0.3


def groq_chat_with_retry(
    client,
    prompt: str,
    *,
    model: str,
    fallback_model: Optional[str] = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> str:
    """Send a chat completion with automatic retry and model fallback.

    Parameters
    ----------
    client          : groq.Groq instance
    prompt          : user message content
    model           : primary LLM model name
    fallback_model  : optional fallback if the primary model fails
    max_tokens      : token limit for the response
    temperature     : sampling temperature
    max_retries     : total attempts before giving up

    Returns
    -------
    str — the LLM response text

    Raises
    ------
    RuntimeError if all retries (including fallback) are exhausted.
    """
    models_to_try = [model]
    if fallback_model and fallback_model != model:
        models_to_try.append(fallback_model)

    last_error: Optional[Exception] = None

    for current_model in models_to_try:
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content

            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()

                # Rate limit — honour Retry-After or back off
                if "429" in str(exc) or "rate" in err_str:
                    sleep_for = min(60.0, 2.0 ** (attempt + 1))
                    print(f"[Groq] 429 rate-limited on {current_model}. "
                          f"Retrying in {sleep_for:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_for)
                    continue

                # Server error — transient, retry
                if "500" in str(exc) or "502" in str(exc) or "503" in str(exc):
                    sleep_for = min(30.0, 2.0 ** attempt)
                    print(f"[Groq] Server error on {current_model}. "
                          f"Retrying in {sleep_for:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_for)
                    continue

                # Model not found / deprecated — skip to fallback immediately
                if "model" in err_str and ("not found" in err_str or "deprecat" in err_str):
                    print(f"[Groq] Model '{current_model}' unavailable: {exc}")
                    break  # break inner loop, try next model

                # Unknown error — still retry with backoff
                sleep_for = min(30.0, 2.0 ** attempt)
                print(f"[Groq] Error on {current_model}: {exc}. "
                      f"Retrying in {sleep_for:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(sleep_for)

    # All retries exhausted
    raise RuntimeError(
        f"Groq chat completion failed after retries on models {models_to_try}: {last_error}"
    )
