"""Measure contradiction detection on SciFact's expert labels.

SciFact (Wadden et al., EMNLP 2020) pairs biomedical claims with evidence
rationales labelled SUPPORT or CONTRADICT by experts. Each item is a sentence
pair, which is exactly the decision the contradiction pipeline must make for
two finding sentences. Positive class: CONTRADICTED.

Components:
* ``nli``: the symmetric NLI contradiction probability used to rank candidates
  (``contradiction_nli.contradiction_probabilities``). Its threshold is chosen
  on the dev split, then applied once to test. AUROC is threshold-free.
* LLM verdicts: the application's structured contradiction prompt with the two
  statements as focus sentences, run on test only with no tuning. Responses are
  checkpointed per model so runs resume.

Domain caveat: SciFact is biomedical; the corpus is computer science. This
validates the components on expert labels, not on CS paper pairs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

BENCHMARK = "evaluation/benchmarks/scifact_support_v1.json"


def load_items(split, path=BENCHMARK):
    items = [i for i in json.loads(Path(path).read_text(encoding="utf-8"))["items"] if i["split"] == split]
    return [{"item_id": i["item_id"], "claim": i["claim"], "evidence": i["passage_text"],
             "gold": i["label"] == "CONTRADICTED"} for i in items]


def binary_report(gold, predicted):
    gold, predicted = np.asarray(gold, bool), np.asarray(predicted, bool)
    tp = int((gold & predicted).sum())
    fp = int((~gold & predicted).sum())
    fn = int((gold & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "accuracy": round(float((gold == predicted).mean()), 4), "tp": tp, "fp": fp, "fn": fn,
            "positives": int(gold.sum()), "count": len(gold)}


def auroc(gold, scores):
    gold, scores = np.asarray(gold, bool), np.asarray(scores, float)
    positives, negatives = scores[gold], scores[~gold]
    if not len(positives) or not len(negatives):
        return None
    greater = (positives[:, None] > negatives[None, :]).sum() + 0.5 * (positives[:, None] == negatives[None, :]).sum()
    return round(float(greater / (len(positives) * len(negatives))), 4)


def best_threshold(gold, scores):
    candidates = sorted(set(np.round(scores, 4)))
    return max(candidates, key=lambda t: (binary_report(gold, np.asarray(scores) >= t)["f1"], -t))


def calibration(gold, predicted, confidence, bins=(0.0, 0.5, 0.7, 0.9, 1.0001)):
    gold, predicted, confidence = map(np.asarray, (gold, predicted, confidence))
    rows, ece = [], 0.0
    correct = gold == predicted
    for low, high in zip(bins, bins[1:]):
        mask = (confidence >= low) & (confidence < high)
        if mask.any():
            accuracy, mean_conf = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += mask.mean() * abs(accuracy - mean_conf)
            rows.append({"range": [low, round(min(high, 1.0), 2)], "count": int(mask.sum()),
                         "accuracy": round(accuracy, 4), "mean_confidence": round(mean_conf, 4)})
    return {"bins": rows, "expected_calibration_error": round(float(ece), 4)}


def llm_verdict(client, claim, evidence):
    from src.pipeline.prompts import build_contradiction_prompt
    from src.pipeline.verdicts import CONTRADICTION_JSON_SCHEMA, parse_contradiction_response

    first = {"title": "Statement A", "year": "n/a", "abstract": claim}
    second = {"title": "Statement B", "year": "n/a", "abstract": evidence}
    prompt = build_contradiction_prompt(first, second, structured=True, abstract_chars=1500,
                                        evidence=({"text": claim}, {"text": evidence}))
    options = {"json_schema": CONTRADICTION_JSON_SCHEMA} if client.provider == "ollama" else {}
    if client.provider == "groq" and "gpt-oss" in client.model:
        options["reasoning_effort"] = "low"
    raw = client.complete(prompt, max_tokens=1500, temperature=0.0, **options)
    return raw, parse_contradiction_response(raw)


def run_llm(client, items, checkpoint: Path):
    done = {}
    if checkpoint.exists():
        # Later rows win, and failed requests (e.g. rate-limit exhaustion) are retried
        # on resume instead of being scored as wrong answers.
        for row in map(json.loads, checkpoint.read_text(encoding="utf-8").splitlines()):
            if row["raw"].startswith("ERROR"):
                done.pop(row["item_id"], None)
            else:
                done[row["item_id"]] = row
    with checkpoint.open("a", encoding="utf-8") as stream:
        for index, item in enumerate(items, 1):
            if item["item_id"] in done:
                continue
            try:
                raw, parsed = llm_verdict(client, item["claim"], item["evidence"])
            except Exception as error:  # record failures as invalid output, never as a verdict
                raw, parsed = f"ERROR {type(error).__name__}", {"verdict": "UNKNOWN", "confidence": None, "valid": False}
            row = {"item_id": item["item_id"], "verdict": parsed["verdict"], "confidence": parsed["confidence"],
                   "valid": parsed["valid"], "raw": raw[:2000]}
            done[item["item_id"]] = row
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if index % 25 == 0:
                print(f"{client.model}: {index}/{len(items)}", flush=True)
    return [done[item["item_id"]] for item in items]


def llm_report(items, rows):
    gold = [item["gold"] for item in items]
    predicted = [row["verdict"] == "CONTRADICTION" for row in rows]
    valid = [row["valid"] for row in rows]
    confidence = [row["confidence"] if row["confidence"] is not None else 0.0 for row in rows]
    verdicts = {}
    for row in rows:
        verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1
    return {"binary": binary_report(gold, predicted), "valid_rate": round(float(np.mean(valid)), 4),
            "verdict_counts": verdicts, "calibration": calibration(gold, predicted, confidence),
            "confidence_distinct_values": sorted({round(c, 2) for c in confidence})[:20]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="results/reasoning/contradiction_verdicts_scifact_20261114")
    parser.add_argument("--llm", nargs="*", default=[], help="ollama:<model> or groq:<model>")
    parser.add_argument("--skip-nli", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    test = load_items("test")
    report_path = output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "benchmark": BENCHMARK, "positive_class": "CONTRADICTED", "test_count": len(test),
        "test_positives": sum(i["gold"] for i in test), "label_source": "SciFact expert annotations"}

    if not args.skip_nli:
        from src.pipeline.contradiction_nli import contradiction_probabilities

        dev = load_items("dev")
        dev_scores = contradiction_probabilities([(i["claim"], i["evidence"]) for i in dev])
        test_scores = contradiction_probabilities([(i["claim"], i["evidence"]) for i in test])
        threshold = best_threshold([i["gold"] for i in dev], dev_scores)
        report["nli"] = {
            "model": "cross-encoder/nli-deberta-v3-small (symmetric mean contradiction probability)",
            "threshold_selected_on_dev": float(threshold),
            "dev_auroc": auroc([i["gold"] for i in dev], dev_scores),
            "test_auroc": auroc([i["gold"] for i in test], test_scores),
            "test_at_dev_threshold": binary_report([i["gold"] for i in test], test_scores >= threshold),
            "test_at_0.5": binary_report([i["gold"] for i in test], test_scores >= 0.5),
        }
        np.save(output / "nli_test_scores.npy", test_scores)
        print("NLI", json.dumps(report["nli"]), flush=True)

    from src.utils.llm_client import GroqLLMClient, OllamaLLMClient
    from src.utils import config

    for spec in args.llm:
        provider, model = spec.split(":", 1)
        if provider == "ollama":
            client = OllamaLLMClient(base_url=config.OLLAMA_BASE_URL, model=model, timeout_sec=300,
                                     keep_alive=config.OLLAMA_KEEP_ALIVE, context_length=8192, max_retries=4)
        else:
            from groq import Groq

            client = GroqLLMClient(Groq(api_key=config.GROQ_API_KEY), model=model, fallback_model=model, max_retries=6)
        started = time.time()
        rows = run_llm(client, test, output / f"llm_{provider}_{model.replace('/', '_').replace(':', '_')}.jsonl")
        report.setdefault("llm", {})[spec] = {**llm_report(test, rows), "seconds": round(time.time() - started)}
        print(spec, json.dumps(report["llm"][spec]["binary"]), "valid", report["llm"][spec]["valid_rate"], flush=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
