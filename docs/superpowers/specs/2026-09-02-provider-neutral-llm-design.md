# Provider-Neutral LLM Design

Date: 2026-09-02
Status: Approved in chat; pending written-spec review
Branch: `phase3`

## Objective

Remove Groq as a production runtime dependency by adding local Ollama support
behind a provider-neutral LLM interface. Ollama will be the configured local
provider for the application, while Groq remains available for reproducibility
and experiments that explicitly compare hosted Groq models.

## Scope

The provider-neutral interface will cover:

- Literature-review generation and structured-output repair.
- Contradiction classification.
- Hypothesis generation.
- The orchestrator and Streamlit resource initialization.
- Reasoning-output collection.
- Claim-support AI evaluation where a general LLM judge is requested.
- Read-only live completion checks.
- Runtime and artifact metadata that identify the active provider and model.

The following remain explicitly Groq-specific because Groq is part of the
experiment definition:

- Controlled Groq LLM model-comparison experiments.
- Existing Groq AI-annotation reproduction commands and frozen artifacts.
- Historical evidence under `teammate2/`.

Retrieval, embeddings, NER, Chroma, Neo4j, provenance parsing, semantic NLI,
and benchmark metric formulas are outside this change.

## Configuration Contract

The primary runtime configuration will be:

```env
LLM_PROVIDER=ollama
LLM_MAX_RETRIES=3

OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:3b-instruct
OLLAMA_FALLBACK_MODEL=
OLLAMA_TIMEOUT_SEC=300
OLLAMA_KEEP_ALIVE=10m
```

Groq remains configurable through:

```env
GROQ_API_KEY=replace_me
LLM_MODEL=openai/gpt-oss-120b
LLM_MODEL_FALLBACK=llama-3.1-8b-instant
GROQ_MAX_RETRIES=4
```

`LLM_PROVIDER` accepts only `ollama` or `groq`. `LLM_MAX_RETRIES` is the common
runtime retry setting; when absent, existing Groq retry configuration remains a
compatibility fallback for Groq. Existing Groq model variables retain their
meaning and are not reinterpreted as Ollama model names.

There is no automatic fallback between providers. In particular, selecting
Ollama can never cause a hidden Groq request.

## Architecture

### Common Interface

Add `src/utils/llm_client.py` as the production boundary. It will expose:

- A small client protocol with `complete(prompt, *, max_tokens, temperature,
  reasoning_effort=None) -> str`.
- An `OllamaLLMClient` implementation.
- A `GroqLLMClient` adapter around the existing retry wrapper.
- A factory that validates configuration and constructs the selected provider.
- Provider/model metadata suitable for Streamlit and result manifests.

Pipelines will receive a generic LLM client and call `complete()`. They will not
import the Groq SDK, inspect provider names, or build provider-specific payloads.

### Ollama Adapter

The Ollama implementation will use the existing `requests` dependency and the
native local endpoints:

- `GET /api/tags` to verify model availability.
- `POST /api/chat` for non-streaming generation.

The chat payload will contain one user message, `stream: false`, the configured
model, `keep_alive`, and options mapping temperature and maximum output tokens.
The adapter will validate that the response contains non-empty message content.

Model availability is checked lazily before the first completion and cached for
that client instance. This avoids a tags request before every generation while
still producing a clear missing-model error at startup/use.

### Groq Adapter

The Groq adapter constructs the SDK client only when Groq is selected. It calls
the existing `groq_chat_with_retry()` implementation so historical retry,
fallback-model, and optional reasoning-effort behavior remains compatible.

The orchestrator must no longer import `groq.Groq` directly. This ensures an
Ollama runtime does not require a valid Groq key or make any Groq request.

## Runtime Data Flow

```text
.env -> config validation -> get_llm_client()
                              |
                    +---------+---------+
                    |                   |
              Ollama adapter       Groq adapter
                    |                   |
               localhost API       Groq SDK/API
                    +---------+---------+
                              |
             review / contradiction / hypothesis
                              |
           existing parsers, provenance, and audits
```

Streamlit caches the selected generic LLM client with the existing Neo4j driver
and Chroma collection. The sidebar displays both provider and model.

## Failure Behavior

The Ollama client will distinguish:

- Connection refusal: Ollama is not running or the base URL is wrong.
- Missing model: show the exact `ollama pull <model>` command.
- Timeout: local generation exceeded the configured timeout.
- Retryable server failure: retry a bounded number of times with backoff.
- Client/configuration error: fail immediately without pointless retries.
- Invalid or empty response: fail clearly and preserve the pipeline's existing
  audit/error behavior.

Retries apply to transport and transient server failures. They do not retry
successful but malformed scientific output indefinitely. Review retains its one
structured-output repair attempt; contradiction and hypothesis retain their
strict parsing and rejection behavior.

Ollama errors never trigger Groq fallback. Groq rate limits remain relevant only
when `LLM_PROVIDER=groq` or an explicitly Groq-specific experiment is run.

## Evaluation Integration

Reasoning-output collection and general support judging will request the common
client from the factory. Their metadata will record:

- Provider.
- Primary model.
- Provider base URL without credentials or sensitive query data.
- Retry/timeout configuration.
- Git commit and existing artifact hashes where already supported.

Live checks will report the selected provider under a generic LLM check and run
a minimal completion. Provider-specific Groq model-list checks remain available
only in explicit Groq reproduction paths.

Frozen AI-reference artifacts are not rewritten. Their reports continue to
identify the models and providers originally used.

## Compatibility And Migration

- Existing callers that directly use `groq_chat_with_retry()` in explicit Groq
  experiments remain supported.
- Production pipeline function parameter names change from `groq_client` to
  `llm_client`; positional calling remains compatible during migration.
- No persisted paper, Chroma, or Neo4j data is modified.
- No model is downloaded automatically. The selected Ollama model must already
  exist or be installed by the operator.
- The installed local `qwen2.5:3b-instruct` model is the documented default for
  this machine. Its smaller size favors local reliability over maximum reasoning
  quality, so benchmark comparisons must identify it explicitly.

## Testing Strategy

Implementation follows test-driven development. Unit tests will first establish:

1. Provider selection creates Ollama without requiring a Groq key.
2. Unknown providers and malformed configuration fail clearly.
3. Ollama model verification accepts installed models and rejects absent ones.
4. Ollama chat payload maps model, prompt, temperature, token limit, and
   `keep_alive` correctly.
5. Retryable local failures retry within the configured bound.
6. Client errors and missing models do not retry unnecessarily.
7. Ollama failure never constructs or calls Groq.
8. Review, contradiction, and hypothesis use the generic `complete()` contract.
9. Streamlit displays selected provider/model metadata.
10. General reasoning collection/live checks use the provider factory while
    explicit Groq experiments remain unchanged.

Verification requires:

- Focused new unit tests passing.
- Full pytest suite passing.
- `compileall` and `pip check` succeeding.
- A real local Ollama tags check.
- A real short completion using `qwen2.5:3b-instruct`.
- A review-path smoke test when Neo4j is available.
- Documentation and `.env.example` consistency checks.

## Documentation Changes

Update `README.md`, `SETUP.md`, `PROJECT_STATUS.md`, `.env.example`, and the
local untracked runbook. Documentation will explain:

- Installing/starting Ollama and pulling a model.
- Selecting Ollama or Groq.
- Verifying the local endpoint and model.
- Starting the UI without a Groq key.
- Expected performance/quality limits of the installed 3B model.
- Troubleshooting connection, model, timeout, and memory errors.
- Which evaluation commands remain intentionally Groq-specific.

## Acceptance Criteria

The change is complete when:

- `LLM_PROVIDER=ollama` runs production generation without a configured Groq
  key and without importing/constructing a Groq client in that runtime path.
- Review, contradiction, and hypothesis use the same generic client contract.
- Ollama errors are actionable and never fall back to Groq.
- Streamlit identifies the active provider/model.
- Existing explicit Groq experiments remain reproducible.
- All automated and local smoke verification passes.
- Tracked documentation matches the final code and the local runbook remains
  untracked.
