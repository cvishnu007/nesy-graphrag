"""Cross-platform, offline reproduction of the packaged reasoning evidence.

Reads immutable inputs under teammate2/ and writes only a new result directory.
Does not call Groq/Ollama, regenerate annotations, or require Chroma/Neo4j.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from src.evaluation.reasoning_benchmark_io import load_reasoning_benchmark, records_for_split
from src.evaluation.phase3_annotation import load_completed_packets

PACKAGE = Path(__file__).resolve().parents[2] / 'teammate2'
BENCHMARK_NAMES = {'contradiction': 'contradiction_pairs.json', 'support': 'claim_support.json', 'hypothesis': 'hypothesis_ratings.json'}


def verify_digest(data: bytes, expected: str) -> str:
    if hashlib.sha256(data).hexdigest() == expected:
        return 'exact'
    lf = data.replace(b'\r\n', b'\n')
    if any(hashlib.sha256(value).hexdigest() == expected for value in (lf, lf.replace(b'\n', b'\r\n'))):
        return 'line_endings_only'
    raise ValueError('Artifact hash mismatch beyond line-ending normalization')


def verify_frozen_benchmarks(package=PACKAGE):
    directory = Path(package) / 'benchmarks'
    manifest = json.loads((directory / 'ai_reference_frozen_manifest.json').read_text(encoding='utf-8-sig'))
    result = {}
    for task, name in BENCHMARK_NAMES.items():
        path = directory / name
        expected = manifest['benchmarks'][task]
        hash_match = verify_digest(path.read_bytes(), expected['sha256'])
        benchmark = load_reasoning_benchmark(path, task)
        counts = {split: len(records_for_split(benchmark, task, split)) for split in ['dev', 'test']}
        if counts != expected['split_counts'] or benchmark['benchmark_version'] != expected['benchmark_version']:
            raise ValueError(f'Frozen benchmark identity mismatch: {task}')
        result[task] = {'count': sum(counts.values()), 'split_counts': counts,
                        'hash_match': hash_match, 'checkout_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def package_audit(package=PACKAGE):
    package = Path(package)
    manifest = json.loads((package / 'MANIFEST.json').read_text(encoding='utf-8-sig'))
    rows = []
    for entry in manifest['files']:
        path = package / entry['path']
        try: status = verify_digest(path.read_bytes(), entry['sha256'])
        except (ValueError, FileNotFoundError): status = 'changed_or_missing_since_packaging'
        rows.append({'path': entry['path'], 'status': status})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new output directory')
    benchmarks = verify_frozen_benchmarks()
    packets = load_completed_packets(PACKAGE / 'annotations/phase3/reviewer_packets')
    if len(packets) != 313 or sum(map(len, packets.values())) != 392:
        raise ValueError('Packaged annotation coverage mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from src.evaluation.reasoning_runner import run_reasoning_evaluation
    base = PACKAGE / 'evaluations/phase5_ai_reference_baseline'
    failures = []
    for split in ['dev', 'test']:
        result = run_reasoning_evaluation(
            tasks=['contradiction','support','hypothesis'], split=split,
            output_dir=args.output_dir / split,
            benchmark_paths={task: PACKAGE / 'benchmarks' / name for task,name in BENCHMARK_NAMES.items()},
            prediction_paths={'contradiction': base / 'predictions' / f'{split}.jsonl',
                              'support': base / 'predictions' / f'support_{split}.jsonl'},
        )
        failures.extend(result['failures'])
    audit = package_audit()
    summary = {'benchmarks': benchmarks, 'unique_annotation_items': len(packets),
               'annotation_assignments': sum(map(len, packets.values())), 'offline_only': True,
               'human_ground_truth': False, 'reference_source': 'AI-generated',
               'failure_count': len(failures), 'historical_package_manifest_audit': audit,
               'note': 'The package manifest predates some committed edits; report drift separately from frozen benchmark integrity.'}
    (args.output_dir / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'historical_package_manifest_audit'}, indent=2))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
