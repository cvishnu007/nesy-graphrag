"""Reproduce the fixed SciFact claim/evidence NLI baseline from cached weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.evaluation.classification_metrics import classification_report

MODEL = 'cross-encoder/nli-deberta-v3-small'
REVISION = 'fa2804872c3b4bd748f38c0185cc85775361e735'
CLASSES = ['SUPPORTED', 'PARTIALLY_SUPPORTED', 'UNSUPPORTED', 'CONTRADICTED']


def evaluate(benchmark_path, output, *, predictor=None, predictions_path=None):
    benchmark_path, output = Path(benchmark_path), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new result directory')
    raw = benchmark_path.read_bytes()
    items = [r for r in json.loads(raw)['items'] if r['split'] == 'test']
    if not items or len({r['item_id'] for r in items}) != len(items):
        raise ValueError('Expected unique, non-empty held-out items')
    protocol = {
        'benchmark_sha256': hashlib.sha256(raw).hexdigest(),
        'source': 'https://github.com/allenai/scifact',
        'source_split': 'official dev', 'local_split': 'test',
        'model': MODEL, 'model_revision': REVISION, 'batch_size': 8,
        'partial_entailment_floor': 0.20,
        'selection': 'predeclared existing default; no tuning on SciFact test labels',
        'task': 'claim/evidence classification, not paper-pair contradiction',
        'macro_classes': ['SUPPORTED', 'CONTRADICTED'],
        'reference_source': 'published SciFact expert annotations; no project-team relabeling',
        'count': len(items), 'inference_rerun': predictions_path is None,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    if predictions_path is None:
        if predictor is None:
            import torch
            from src.evaluation.support_ai_evaluator import generate_local_nli_predictions
            torch.set_num_threads(2)
            predictor = generate_local_nli_predictions
        visible = [{key: row[key] for key in ['item_id', 'claim', 'passage_text']} for row in items]
        predictions = predictor(visible, output/'predictions.jsonl', model_name=MODEL,
                                model_revision=REVISION, batch_size=8, partial_entailment_floor=.20)
    else:
        prediction_bytes = Path(predictions_path).read_bytes()
        predictions = [json.loads(line) for line in prediction_bytes.splitlines() if line.strip()]
        (output/'prediction_source.json').write_text(json.dumps({
            'sha256': hashlib.sha256(prediction_bytes).hexdigest(),
            'note': 'Offline scoring of saved predictions; model identity is declared, not verified by rerunning inference.'}, indent=2)+'\n')
    indexed = {r['item_id']: r for r in predictions}
    if len(indexed) != len(predictions) or set(indexed) != {r['item_id'] for r in items}:
        raise ValueError('Prediction IDs must exactly match the held-out benchmark')
    for row in predictions:
        row['reference_source'] = 'published SciFact expert annotations'
    (output/'predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in predictions))
    report = classification_report([r['label'] for r in items],
        [indexed[r['item_id']]['prediction'] for r in items], CLASSES,
        macro_classes=['SUPPORTED', 'CONTRADICTED'])
    (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=Path('evaluation/benchmarks/scifact_support_v1.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, help='Score saved predictions without rerunning inference')
    args = parser.parse_args(argv)
    report = evaluate(args.benchmark, args.output_dir, predictions_path=args.predictions)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
