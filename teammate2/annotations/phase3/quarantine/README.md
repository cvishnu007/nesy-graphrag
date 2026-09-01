# Phase 3 invalid-response quarantine

`reviewer_01_support_dev_invalid_wrong_task_responses.json` is a byte-for-byte
copy of the packet found during AI-reference annotation resume. All 42 populated
responses used the contradiction-only label `DIFFERENT SCOPE` and a `reason`
field, so none satisfied the claim-support response schema. The active packet's
42 invalid response fields were reset to `null` and regenerated through the
declared AI-reference annotator. This quarantine is audit evidence only and is
not an annotation source, human review artifact, or benchmark input.
