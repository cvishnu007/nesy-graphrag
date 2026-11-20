# Provider-Neutral LLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run all production generation and general live-evaluation workflows through local Ollama or Groq without making Groq a required production dependency.

**Architecture:** Add one provider-neutral `LLMClient.complete()` boundary with Ollama and Groq implementations selected by configuration. Migrate production and general evaluation callers to that boundary while preserving explicitly Groq-defined comparison and annotation experiments.

**Tech Stack:** Python 3.11, `requests`, Groq SDK, Ollama native HTTP API, pytest, Streamlit.

**Spec:** `docs/superpowers/specs/2026-09-02-provider-neutral-llm-design.md`

## Global Constraints

- `LLM_PROVIDER` accepts only `ollama` or `groq`.
- Selecting Ollama must never instantiate, import at runtime, or call a Groq client.
- There is no automatic cross-provider fallback.
- Use existing `requests`; do not add an Ollama Python dependency.
- Default local model is the already-installed `qwen2.5:3b-instruct`.
- Do not download models automatically or modify Chroma/Neo4j/paper data.
- Keep explicitly Groq-focused LLM comparison and AI annotation experiments Groq-specific.
- Preserve strict parsers, provenance checks, and the single review repair attempt.
- Follow test-driven development: every production behavior starts with a failing test.
- Keep `LOCAL_PROJECT_RUNBOOK.md` untracked.

## File Structure

- Create `src/utils/llm_client.py`: provider protocol, Ollama implementation, Groq adapter, factory, and runtime metadata.
- Create `tests/test_llm_client.py`: provider/configuration, HTTP payload, retry, and no-cross-provider-fallback tests.
- Modify `src/utils/config.py`: typed common/Ollama settings and validation.
- Modify `src/pipeline/review.py`, `contradiction.py`, and `hypothesis.py`: consume `LLMClient.complete()`.
- Modify `src/pipeline/orchestrator.py`: cache a generic client and expose compatibility alias only where needed.
- Modify `app/streamlit_app.py`: cache/display generic provider metadata.
- Modify general evaluation collection/check modules to use the factory.
- Preserve `src/utils/groq_client.py` for explicit Groq workflows.
- Update tracked setup/status documentation and the untracked local runbook.

---

### Task 1: Common Client, Configuration, And Ollama Transport

**Files:**
- Create: `src/utils/llm_client.py`
- Create: `tests/test_llm_client.py`
- Modify: `src/utils/config.py`

**Interfaces:**
- Produces: `LLMClient.complete(prompt: str, *, max_tokens: int, temperature: float, reasoning_effort: str | None = None) -> str`.
- Produces: `OllamaLLMClient`, `GroqLLMClient`, `create_llm_client()`, and `llm_runtime_metadata()`.
- Consumes: existing `groq_chat_with_retry()` for Groq behavior.

- [ ] **Step 1: Write failing configuration and factory tests**

Add tests that monkeypatch `src.utils.llm_client` configuration values and prove:

```python
def test_factory_creates_ollama_without_groq_key(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm_client, "GROQ_API_KEY", None)
    client = llm_client.create_llm_client()
    assert isinstance(client, llm_client.OllamaLLMClient)


def test_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "unknown")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        llm_client.create_llm_client()
```

- [ ] **Step 2: Run factory tests and verify RED**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_llm_client.py -q`

Expected: FAIL because `src.utils.llm_client` does not exist.

- [ ] **Step 3: Add typed configuration**

Add these settings to `src/utils/config.py`:

```python
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
if LLM_PROVIDER not in {"ollama", "groq"}:
    raise ValueError("LLM_PROVIDER must be 'ollama' or 'groq'")

LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", os.getenv("GROQ_MAX_RETRIES", "3")))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct").strip()
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "").strip()
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "300"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m").strip()
```

Validate non-empty URL/model/keep-alive and positive retry/timeout values.

- [ ] **Step 4: Implement the minimal protocol and factory**

Define:

```python
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
    ) -> str: ...
```

`create_llm_client()` must branch on `LLM_PROVIDER`; import and instantiate the
Groq SDK only inside the Groq branch after validating `GROQ_API_KEY`.

- [ ] **Step 5: Run factory tests and verify GREEN**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_llm_client.py -q`

Expected: factory tests PASS.

- [ ] **Step 6: Write failing Ollama payload/model tests**

Use a fake `requests.Session` and assert:

```python
assert posted_url == "http://127.0.0.1:11434/api/chat"
assert payload["model"] == "qwen2.5:3b-instruct"
assert payload["messages"] == [{"role": "user", "content": "prompt"}]
assert payload["stream"] is False
assert payload["keep_alive"] == "10m"
assert payload["options"] == {"temperature": 0.0, "num_predict": 300}
```

Also assert `/api/tags` is checked once, missing models produce an error containing
`ollama pull qwen2.5:3b-instruct`, and a second completion does not repeat the tags
request.

- [ ] **Step 7: Run Ollama tests and verify RED**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_llm_client.py -q`

Expected: FAIL because Ollama HTTP behavior is not implemented.

- [ ] **Step 8: Implement Ollama model verification and completion**

Use `Session.get(.../api/tags, timeout=...)`, `Session.post(.../api/chat,
json=payload, timeout=...)`, `raise_for_status()`, and validate non-empty
`response.json()["message"]["content"]`.

- [ ] **Step 9: Write failing retry and isolation tests**

Prove retryable connection/server failures retry at most `LLM_MAX_RETRIES`, an
HTTP 4xx model/configuration failure is immediate, and monkeypatched Groq
construction remains untouched when every Ollama attempt fails.

- [ ] **Step 10: Implement bounded Ollama retry behavior**

Retry `requests.ConnectionError`, `requests.Timeout`, and HTTP 5xx with bounded
exponential sleep. Convert final failures into actionable `RuntimeError` messages.
Do not retry other HTTP 4xx responses or invalid response schemas.

- [ ] **Step 11: Run Task 1 tests**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_llm_client.py tests/test_groq_client.py -q`

Expected: all tests PASS.

- [ ] **Step 12: Commit Task 1**

```powershell
git add src/utils/config.py src/utils/llm_client.py tests/test_llm_client.py
git commit -m "Add provider-neutral Ollama client"
```

---

### Task 2: Production Review, Contradiction, And Hypothesis Migration

**Files:**
- Modify: `src/pipeline/review.py`
- Modify: `src/pipeline/contradiction.py`
- Modify: `src/pipeline/hypothesis.py`
- Modify: `tests/test_provenance.py`
- Modify: `tests/test_verdicts.py`
- Modify: `tests/test_hypotheses.py`

**Interfaces:**
- Consumes: Task 1 `LLMClient.complete()`.
- Produces: unchanged result dictionaries and parsers with provider-independent generation.

- [ ] **Step 1: Replace Groq mocks with a fake generic client in tests**

Use:

```python
class FakeLLMClient:
    provider = "test"
    model = "fixture"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, prompt, **options):
        self.calls.append({"prompt": prompt, **options})
        return next(self.responses)
```

Add assertions proving each workflow calls `complete()` with its existing
temperature and token limit.

- [ ] **Step 2: Run focused pipeline tests and verify RED**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_provenance.py tests/test_verdicts.py tests/test_hypotheses.py -q`

Expected: FAIL because production functions still call `groq_chat_with_retry()`.

- [ ] **Step 3: Migrate production functions**

Rename function parameters to `llm_client` and replace every generation/repair
call with:

```python
llm_client.complete(
    prompt,
    max_tokens=existing_limit,
    temperature=0.0,
)
```

Remove Groq model/retry imports from these pipeline modules. Preserve exception
capture, parsing, repair, provenance, and rejected-output behavior.

- [ ] **Step 4: Run focused pipeline tests and verify GREEN**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_provenance.py tests/test_verdicts.py tests/test_hypotheses.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/pipeline/review.py src/pipeline/contradiction.py src/pipeline/hypothesis.py tests/test_provenance.py tests/test_verdicts.py tests/test_hypotheses.py
git commit -m "Use generic LLM client in reasoning pipelines"
```

---

### Task 3: Orchestrator And Streamlit Integration

**Files:**
- Modify: `src/pipeline/orchestrator.py`
- Modify: `app/streamlit_app.py`
- Create: `tests/test_orchestrator_llm.py`
- Modify: `tests/test_streamlit_metrics.py`

**Interfaces:**
- Consumes: `create_llm_client()` and `llm_runtime_metadata()`.
- Produces: `get_llm()` cached runtime client; `get_groq()` may remain only as a deprecated compatibility alias when provider is Groq.

- [ ] **Step 1: Write failing lazy-factory and UI metadata tests**

Assert that two `get_llm()` calls return one cached fake object, Ollama selection
does not require `GROQ_API_KEY`, and AST/source inspection finds sidebar output
for both `LLM_PROVIDER` and the active model instead of a Groq-only label.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_orchestrator_llm.py tests/test_streamlit_metrics.py -q`

Expected: FAIL because `get_llm()` does not exist and Streamlit calls `get_groq()`.

- [ ] **Step 3: Implement generic resource caching**

Replace `_groq_client/get_groq()` production use with `_llm_client/get_llm()`.
Pass the generic client to all three workflows. In Streamlit, rename local
variables to `llm_client`, cache the generic resource, and display:

```text
LLM: ollama / qwen2.5:3b-instruct
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_orchestrator_llm.py tests/test_streamlit_metrics.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/pipeline/orchestrator.py app/streamlit_app.py tests/test_orchestrator_llm.py tests/test_streamlit_metrics.py
git commit -m "Select configured LLM provider at runtime"
```

---

### Task 4: General Evaluation And Live Checks

**Files:**
- Modify: `src/evaluation/collect_reasoning_outputs.py`
- Modify: `src/evaluation/live_completion_checks.py`
- Modify: `src/evaluation/support_ai_evaluator.py`
- Modify: `src/evaluation/reasoning_runner.py`
- Modify: `src/evaluation/artifact_manifest.py`
- Modify: `tests/test_live_completion_checks.py`
- Modify: `tests/test_support_ai_evaluator.py`
- Modify: `tests/test_reasoning_end_to_end.py`
- Modify: `tests/test_phase1_phase2_tooling.py`

**Interfaces:**
- Consumes: generic client factory and runtime metadata.
- Produces: provider-labeled general evaluation artifacts.
- Preserves: `src/evaluation/llm_comparison.py` and `phase3_ai_annotation.py` as explicit Groq experiments.

- [ ] **Step 1: Write failing provider-neutral evaluation tests**

Assert general collectors receive a fake generic client, live checks return an
`llm` result containing provider/model, support judging calls `complete()`, and
prerequisite metadata reports selected-provider readiness rather than requiring
Groq unconditionally.

- [ ] **Step 2: Run focused evaluation tests and verify RED**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_live_completion_checks.py tests/test_support_ai_evaluator.py tests/test_reasoning_end_to_end.py tests/test_phase1_phase2_tooling.py -q`

Expected: FAIL at Groq-specific production/general-evaluation construction.

- [ ] **Step 3: Migrate general evaluation paths**

Use `create_llm_client()` for collection, support judging, and selected-provider
live completion. Record `provider`, `model`, sanitized base URL, timeout, and
retry count where metadata is already emitted. Do not modify frozen artifacts or
the explicitly Groq comparison/annotation modules.

- [ ] **Step 4: Run focused evaluation tests and verify GREEN**

Run the Step 2 command again. Expected: PASS.

- [ ] **Step 5: Run explicit Groq regression tests**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_groq_client.py tests/test_llm_comparison.py tests/test_phase3_ai_annotation.py -q`

Expected: PASS, proving historical Groq workflows remain supported.

- [ ] **Step 6: Commit Task 4**

```powershell
git add src/evaluation/collect_reasoning_outputs.py src/evaluation/live_completion_checks.py src/evaluation/support_ai_evaluator.py src/evaluation/reasoning_runner.py src/evaluation/artifact_manifest.py tests/test_live_completion_checks.py tests/test_support_ai_evaluator.py tests/test_reasoning_end_to_end.py tests/test_phase1_phase2_tooling.py
git commit -m "Use configured LLM in general evaluation workflows"
```

---

### Task 5: Configuration And Operator Documentation

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `SETUP.md`
- Modify: `PROJECT_STATUS.md`
- Modify locally only: `LOCAL_PROJECT_RUNBOOK.md`
- Test: `tests/test_core_guards.py`

**Interfaces:**
- Documents the exact Task 1 configuration contract and Task 4 scope boundary.

- [ ] **Step 1: Add failing configuration-default guards**

Assert the default provider/model/base URL and positive timeout/retry settings.
Assert invalid providers fail during a clean config import or through a dedicated
validation helper, following the existing configuration-test pattern.

- [ ] **Step 2: Run configuration tests and verify RED**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_core_guards.py -q`

Expected: FAIL until final configuration names/defaults match the contract.

- [ ] **Step 3: Finalize `.env.example` and documentation**

Document these operator commands:

```powershell
ollama serve
ollama list
ollama pull qwen2.5:3b-instruct
Invoke-RestMethod http://127.0.0.1:11434/api/tags
.\venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

State that Groq is optional for normal application use, identify explicit
Groq-only evaluation commands, explain the 3B model quality limitation, and add
connection/model/timeout/memory troubleshooting. Update project status from
Groq-hosted production generation to provider-neutral local/hosted generation.

- [ ] **Step 4: Update the local runbook without tracking it**

Add provider selection, Ollama startup, smoke completion, and recovery commands.
Verify `git ls-files LOCAL_PROJECT_RUNBOOK.md` returns no path.

- [ ] **Step 5: Run configuration tests and verify GREEN**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_core_guards.py -q`

Expected: PASS.

- [ ] **Step 6: Commit tracked documentation**

```powershell
git add .env.example README.md SETUP.md PROJECT_STATUS.md tests/test_core_guards.py
git commit -m "Document local Ollama runtime setup"
```

Do not add `LOCAL_PROJECT_RUNBOOK.md`.

---

### Task 6: Full And Live Verification

**Files:**
- Modify only if verification exposes a defect in files already owned by Tasks 1-5.

**Interfaces:**
- Verifies every acceptance criterion from the design specification.

- [ ] **Step 1: Run complete automated verification**

```powershell
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m compileall -q src app tests
.\venv\Scripts\python.exe -m pip check
git diff --check
```

Expected: all tests pass, compilation/pip/diff checks exit zero, with only known
non-fatal dependency warnings documented.

- [ ] **Step 2: Verify the local Ollama service and installed model**

```powershell
ollama list
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

Expected: `qwen2.5:3b-instruct` appears in both checks.

- [ ] **Step 3: Run a real provider-client smoke completion**

```powershell
$env:LLM_PROVIDER='ollama'
.\venv\Scripts\python.exe -c "from src.utils.llm_client import create_llm_client; c=create_llm_client(); print(c.complete('Reply with exactly: OLLAMA_OK', max_tokens=20, temperature=0.0))"
```

Expected: non-empty local response containing `OLLAMA_OK`, with no Groq key or
network request required.

- [ ] **Step 4: Run a production review smoke test when Neo4j is available**

```powershell
$env:LLM_PROVIDER='ollama'
.\venv\Scripts\python.exe -c "from src.pipeline.orchestrator import graphrag_query; r=graphrag_query('database query optimization', mode='review', top_k=3); print(r['provenance']['stats']); print(r['answer'])"
```

Expected: retrieval completes, Ollama returns structured output or one repair is
audited, and no Groq rate-limit message appears. If Neo4j is stopped, record the
smoke test as blocked by service state rather than claiming it passed.

- [ ] **Step 5: Confirm repository boundaries**

Run:

```powershell
git status --short --branch
git ls-files LOCAL_PROJECT_RUNBOOK.md
git diff --name-only origin/phase3...HEAD
```

Expected: local runbook is untracked; no paper/store artifacts changed; only
planned code, tests, configuration, and tracked docs are included.

- [ ] **Step 6: Commit any verification fixes and push**

If Step 1-5 required corrections, commit them with a focused message. Then:

```powershell
git push origin phase3
```

Report exact test counts, live-smoke evidence, commit hashes, and any blocked
Neo4j-only verification.
