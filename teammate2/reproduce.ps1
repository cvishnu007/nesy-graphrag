# Run from the NeSy-GraphRAG repository root.
# Benchmarks and reasoning annotations are AI-generated references, not human ground truth.

.\venv\Scripts\python.exe -m pytest teammate2/tests -q
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m compileall -q src tests
.\venv\Scripts\python.exe -m pip check

# Validate the completed reviewer packets without changing them.
Get-ChildItem evaluation/phase3/reviewer_packets -Recurse -Filter *.json |
    ForEach-Object {
        .\venv\Scripts\python.exe -m src.evaluation.phase3_annotation `
            validate-packet --packet $_.FullName --require-complete
    }

# Authorized read-only live integration gate. Requires configured Groq and a running Neo4j database.
.\venv\Scripts\python.exe -m src.evaluation.live_completion_checks `
    --output-dir results/reasoning/final_ai_reference/live_checks `
    --external-authorized
