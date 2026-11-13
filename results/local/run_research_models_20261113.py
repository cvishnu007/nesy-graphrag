"""Chain: SciERC NER selection -> corpus extraction -> Neo4j load -> OpenScholar benchmarks."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
PY = str(ROOT / "venv/Scripts/python.exe")
ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}
LOG = Path("data/external_sci_models_download.log")


def run(args, log):
    print("RUN", " ".join(args), flush=True)
    with open(log, "w", encoding="utf-8") as stream:
        code = subprocess.call([PY, "-u", *args], stdout=stream, stderr=subprocess.STDOUT, env=ENV)
    print("EXIT", code, log, flush=True)
    if code:
        sys.exit(code)


while "DONE" not in LOG.read_text(encoding="utf-8", errors="ignore"):
    time.sleep(20)
text = LOG.read_text(encoding="utf-8", errors="ignore")
ner_models = [m for m in ("anlp/SciERC_trained_Scibert", "roccas98/Scibert_finetuned_scierc", "anlp/SciERC_trained_SciDeBERTa")
              if f"OK {m}" in text]
print("available NER models:", ner_models, flush=True)

report_path = "results/external/scierc_ner_20261113.json"
run(["-m", "src.ingestion.scientific_ner", "evaluate", "--models", *ner_models, "spacy", "--output", report_path],
    "results/local/scierc_ner_eval_20261113.log")
report = json.loads(Path(report_path).read_text())
best = max(ner_models, key=lambda m: report["models"][m]["typed"]["f1"])
print("selected NER:", best, report["models"][best]["typed"]["f1"], flush=True)
Path("results/external/scierc_ner_selection_20261113.json").write_text(json.dumps(
    {"selected": best, "criterion": "typed exact-span F1 on SciERC test", "scores":
     {m: report["models"][m]["typed"]["f1"] for m in ner_models}}, indent=2) + "\n")

run(["-m", "src.ingestion.scientific_ner", "extract", "--model", best, "--output", "data/s2_ner_scientific.json"],
    "results/local/scientific_ner_extract_20261113.log")
run(["-m", "src.storage.graph_enrichment", "scientific-entities", "--write",
     "--report", "results/local/graph_enrichment_scientific_entities_20261113.json"],
    "results/local/graph_enrichment_scientific_entities_20261113.log")

if "OK OpenSciLM/OpenScholar_Retriever" in text:
    for bench in ("scidocs", "litsearch"):
        run(["-m", "src.evaluation.external_benchmarks", bench, "--models", "specter", "bge_base", "openscholar",
             "--output-dir", f"results/external/{bench}_openscholar_20261113"],
            f"results/local/external_{bench}_openscholar_20261113.log")
print("CHAIN DONE", flush=True)
