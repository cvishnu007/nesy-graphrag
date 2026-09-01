# AI-reference evaluation reproduction commands. No human ground truth is used.
.\venv\Scripts\python.exe -m pytest tests/test_freeze_ai_reference_benchmarks.py tests/test_contradiction_candidate_evaluator.py tests/test_support_baseline_comparison.py tests/test_hypothesis_metrics.py tests/test_ner_comparison.py tests/test_embedding_comparison.py tests/test_llm_comparison.py tests/test_groq_client.py tests/test_live_completion_checks.py -q
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m compileall -q src tests
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m src.evaluation.live_completion_checks --output-dir results/reasoning/final_ai_reference/live_checks --external-authorized
