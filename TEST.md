## Step 1 — Run the OLD (pre-fix) pipeline and save its output

You're checked out on the old commit right now. Run the full pipeline once:

```bash
python -m src.pipeline.orchestrator
```

This calls `graphrag_query()` in review, contradict, and hypothesis modes and prints/returns results. Save these three things to disk before switching branches:

```bash
mkdir -p fixtures/before
```

In `orchestrator.py`'s `__main__` block (or a quick throwaway script), dump:

```python
import json
json.dump(r1, open("fixtures/before/review_result.json", "w"), indent=2, default=str)
json.dump(r2, open("fixtures/before/contradiction_result.json", "w"), indent=2, default=str)
json.dump(scores, open("fixtures/before/metrics.json", "w"), indent=2, default=str)
```

Where `r1` = review result, `r2` = contradiction result, `scores` = the combined `compute_all_metrics(r1, contradiction_result=r2)` output.

**Use the exact same test query both times** — the codebase's own test query is `"graph neural networks for node classification"`. Keep it identical across old and new runs or the comparison is meaningless.

**What specifically to check/save from this old run**, since that's what you're diffing against:
- `metrics.json` → `nbr.nbr` value (should show the inflated 1.0 bug)
- `metrics.json` → `rdi.rdi` value (should show the false-positive-inflated number, e.g. the documented 0.0667 instead of the correct 0.0)
- `review_result.json` → `papers[*].source` field (to manually inspect the mistagging — this is your evidence for bug #2)
- `contradiction_result.json` → each pair's `llm_analysis` text (to find the negated-verdict case that RDI wrongly counted)

## Step 2 — Switch back to your current (fixed) branch

```bash
git checkout <your-branch-with-all-11-changes>
```

Run the identical query through the identical entry point:

```bash
python -m src.pipeline.orchestrator
```

Save to `fixtures/after/` with the same three filenames.

## Step 3 — Diff

What you're actually looking for, side by side:

| Check | Old (`before/`) | New (`after/`) |
|---|---|---|
| `nbr.nbr` | inflated (~1.0 regardless of retrieval mix) | should now vary meaningfully based on actual graph vs. neural split |
| `nbr.graph_count` / `nbr.total` | should match `1.0` suspiciously often | should reflect real counts |
| `papers[*].source` distribution | mislabeled — check for `"both"` on papers that shouldn't be | correctly split between `"neural"`/`"symbolic"`/`"both"` |
| `rdi.rdi` | documented ~0.0667 (false positive included) | should drop to the correct value (e.g. 0.0 if genuinely no contradictions) |
| `rdi.contradictions_resolved` | counts a negated verdict as resolved | should not count "does NOT constitute a CONTRADICTION" as resolved |
| Streamlit render / any crash | — | confirm #7/#11 refactor didn't break anything |
| Fabricated ID test | n/a (not part of orchestrator test query) | run manually — confirm validator still blocks it after #4/#5 error-handling changes |

A simple diff script:

```python
import json

before = json.load(open("fixtures/before/metrics.json"))
after  = json.load(open("fixtures/after/metrics.json"))

for metric in ["ts", "nbr", "atd", "rdi"]:
    print(f"{metric.upper()}: before={before[metric][metric]}  after={after[metric][metric]}")
```

That gives you a clean, reproducible before/after table you can drop straight into the dissertation's Phase 3 evaluation section as evidence the two bug fixes actually did something — which is exactly the artifact you were worried about losing.