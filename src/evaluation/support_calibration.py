"""Dev-only calibration of partial support using frozen, saved NLI probabilities.

This is a new post-hoc baseline experiment, not an independent replication of
previously published results. No inference, corpus mutation, or label rewriting.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from src.evaluation.claim_support_metrics import evaluate_claim_support
from src.evaluation.semantic_support import nli_scores_to_prediction

FLOORS = (.005, .01, .02, .05, .10, .15, .20, .30, .40)


def remap_predictions(rows, *, floor):
    if not math.isfinite(floor) or not 0 <= floor <= 1:
        raise ValueError('floor must be in [0, 1]')
    predictions, seen = [], set()
    for row in rows:
        identity = row['item_id']
        if identity in seen:
            raise ValueError('Duplicate prediction ID')
        seen.add(identity)
        scores = row['nli_scores']
        if (set(scores) != {'entailment', 'neutral', 'contradiction'} or
            any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()) or
            not math.isclose(sum(scores.values()), 1., abs_tol=.001)):
            raise ValueError('NLI scores must be finite normalized probabilities')
        decision = nli_scores_to_prediction(scores, partial_entailment_floor=floor)
        predictions.append({'item_id': identity, 'prediction': decision['label'],
                            'confidence': decision['confidence'], 'valid': True,
                            'nli_scores': scores})
    return predictions


def select_partial_floor(dev_rows, *, floors=FLOORS):
    rows = list(dev_rows)
    if not rows or any(row.get('split') != 'dev' for row in rows):
        raise ValueError('Threshold selection requires only non-empty dev data')
    if not floors:
        raise ValueError('At least one candidate floor required')
    sweep = []
    for floor in floors:
        predictions = remap_predictions(rows, floor=floor)
        metrics = evaluate_claim_support([{**row, **prediction} for row, prediction in zip(rows, predictions)])
        sweep.append({'floor': floor, 'metrics': metrics})
    # Tie-break toward the more conservative floor, never inspect test labels.
    best = max(sweep, key=lambda row: (row['metrics']['macro_f1'], row['floor']))
    return {'selection_split': 'dev', 'partial_entailment_floor': best['floor'],
            'primary_metric': 'macro_f1', 'tie_break': 'larger floor', 'sweep': sweep}


def coverage_report(rows, *, minimum_per_class=10):
    if minimum_per_class < 1:
        raise ValueError('minimum_per_class must be positive')
    classes = ('CONTRADICTION', 'AGREEMENT', 'DIFFERENT SCOPE')
    records = list(rows)
    counts = {split: dict(Counter(r['label'] for r in records if r['split'] == split)) for split in ('dev', 'test')}
    needed = {split: {label: max(0, minimum_per_class - counts[split].get(label, 0)) for label in classes} for split in counts}
    return {'counts': counts, 'minimum_per_class': minimum_per_class, 'additional_required': needed,
            'ready_for_class_comparison': not any(n for split in needed.values() for n in split.values()),
            'interpretation': 'Engineering coverage threshold, not a statistical power guarantee; add a new independent benchmark, never insert examples into a frozen test set.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    base = root / 'teammate2/evaluations/final_ai_reference/support_comparison'
    parser.add_argument('--dev-predictions', type=Path, default=base / 'dev/semantic_predictions.jsonl')
    parser.add_argument('--test-predictions', type=Path, default=base / 'test/semantic_predictions.jsonl')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    dev_bytes = args.dev_predictions.read_bytes()
    dev = [json.loads(line) for line in dev_bytes.splitlines() if line.strip()]
    selection = select_partial_floor(dev)
    selection.update({'dev_sha256': hashlib.sha256(dev_bytes).hexdigest(), 'human_ground_truth': False,
                      'reference_annotation_source': 'AI-generated', 'experiment': 'post-hoc dev-only partial-support calibration'})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Persist the choice before even loading test predictions.
    (args.output_dir / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    test_bytes = args.test_predictions.read_bytes()
    test = [json.loads(line) for line in test_bytes.splitlines() if line.strip()]
    if not test or any(row.get('split') != 'test' for row in test):
        raise ValueError('Expected non-empty test-only prediction file')
    if {r['item_id'] for r in dev} & {r['item_id'] for r in test}:
        raise ValueError('Dev/test overlap')
    reports = {}
    for name, floor in [('original', .20), ('dev_calibrated', selection['partial_entailment_floor'])]:
        predictions = remap_predictions(test, floor=floor)
        merged = [{**row, **prediction} for row, prediction in zip(test, predictions)]
        reports[name] = evaluate_claim_support(merged)
        (args.output_dir / f'{name}_predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in predictions))
    report = {'test_sha256': hashlib.sha256(test_bytes).hexdigest(), 'test_count': len(test),
              'selected_floor': selection['partial_entailment_floor'], 'human_ground_truth': False,
              'reference_annotation_source': 'AI-generated', 'reports': reports}
    (args.output_dir / 'test_metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    from src.evaluation.reasoning_benchmark_io import load_reasoning_benchmark, records_for_split
    benchmark = load_reasoning_benchmark(root / 'teammate2/benchmarks/contradiction_pairs.json', 'contradiction')
    pairs = records_for_split(benchmark, 'contradiction', 'dev') + records_for_split(benchmark, 'contradiction', 'test')
    (args.output_dir / 'contradiction_coverage.json').write_text(json.dumps(coverage_report(pairs), indent=2) + '\n')
    print(json.dumps({name: {'macro_f1': m['macro_f1'], 'partial_f1': m['per_class']['PARTIALLY_SUPPORTED']['f1'], 'false_acceptance': m['false_acceptance_rate']} for name, m in reports.items()}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
