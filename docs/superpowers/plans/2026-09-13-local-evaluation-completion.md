# Local demo and evaluation completion

User approved the five recommended workstreams on September 13, 2026 and requested continuation.

Design: reuse the September 2 provider-neutral LLM specification, preserve frozen historical evidence, prepare new corpus-specific retrieval pools before grading, and score saved rankings without test tuning. Add BM25 and a conventional corpus-validated vector RAG arm using the same generator, prompt, and explicit character/output budgets as GraphRAG. Keep AI-reference conclusions separate from external published labels and new human review.

- [x] Create branch `feat/local-evaluation` from `origin/phase3`; baseline 250 tests.
- [x] Implement Ollama client and generic production/evaluation consumers; maintain explicit Groq experiments.
- [x] Install Ollama and Neo4j, create a dedicated project database, configure ignored `.env`.
- [x] Download local qwen2.5:3b-instruct and verify real completion.
- [x] Preserve original JSON artifacts; finish 11-topic ingestion: 52,820 raw / 47,627 clean, all original IDs retained.
- [x] Finish incremental NER and Chroma indexing; rebuild only dedicated Neo4j and verify exact ID parity.
- [x] Fix default reasoning benchmark paths and add offline cross-platform package reproduction.
- [x] Add deterministic BM25 and production-filtered retrieval to optional pool/evaluation methods.
- [x] Add matched-budget vector/graph RAG comparison with raw output, provenance, context hashes, and frozen protocol.
- [ ] Freeze expanded method-hidden candidate pool; produce explicitly AI-generated reference judgments; evaluate cached dev/test rankings and paired significance.
- [x] Run development-only partial-support calibration using saved NLI scores; retain both improvements and false-acceptance tradeoff.
- [x] Import a separate SciFact rationale benchmark, with source train -> local dev and source dev -> local test, preserving full rationale sets and excluding duplicate content.
- [x] Evaluate fixed NLI on SciFact and record contradiction-class metrics separately from paper-pair detection.
- [ ] Run all local application modes and matched RAG comparison; start Streamlit.
- [ ] Review changes, run full tests/compile/pip/diff checks, update authoritative status/setup docs, commit locally and expose branch in original workspace.

Rulings: no repeated design approval because the user explicitly authorized these workstreams. No generated label is called a new human judgment. Corpus expansion counts differ from the teammate snapshot because this is a fresh upstream fetch. GPU NER competed with SPECTER on the 8 GB Mac, so use two CPU workers for NER. The agent assigned provider migration stopped at a service usage limit; the primary agent owns finishing and reviewing its changes. No shared-branch push is required.
