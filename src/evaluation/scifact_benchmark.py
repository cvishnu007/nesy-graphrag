"""Import SciFact rationales as a separate external claim-evidence benchmark.

Official train is used for development; official dev is held out for testing.
This measures claim/evidence entailment, not graph candidate discovery or
paper-to-paper contradiction. No labels are inferred from unlabelled claims.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

SOURCE = 'https://github.com/allenai/scifact'


def convert_claims(claims, corpus, *, split, source_split):
    if split not in {'dev','test'}:
        raise ValueError('split must be dev or test')
    rows, seen, contents = [], set(), {}
    for claim in claims:
        for doc_id, evidence_sets in sorted(claim['evidence'].items()):
            if doc_id not in corpus:
                raise ValueError(f'Missing corpus document {doc_id}')
            document = corpus[doc_id]
            for evidence in evidence_sets:
                source_label = evidence['label']
                if source_label not in {'SUPPORT','CONTRADICT'}:
                    raise ValueError('Unknown source evidence label')
                indices = sorted(set(evidence['sentences']))
                if not indices or any(not isinstance(i,int) or i < 0 or i >= len(document['abstract']) for i in indices):
                    raise ValueError('Invalid evidence sentence reference')
                key = f"{claim['id']}:{doc_id}:" + ','.join(map(str,indices))
                if key in seen:
                    raise ValueError('Duplicate evidence set')
                seen.add(key)
                identity = (claim['claim'].casefold().strip(), doc_id, tuple(indices))
                if identity in contents:
                    if contents[identity] != source_label:
                        raise ValueError('Conflicting source labels for duplicate content')
                    continue
                contents[identity] = source_label
                rows.append({'item_id': 'SF-' + hashlib.sha256(key.encode()).hexdigest()[:16],
                    'split':split, 'source_split':source_split, 'query_id':f"SF-claim-{claim['id']}",
                    'claim':claim['claim'], 'passage_id':f"SF-{doc_id}-" + '-'.join(map(str,indices)),
                    'passage_text':' '.join(document['abstract'][i] for i in indices),
                    'paper_id':f'SF-{doc_id}', 'label':'SUPPORTED' if source_label == 'SUPPORT' else 'CONTRADICTED',
                    'notes':'Label mapped from the published SciFact annotation; complete rationale kept together.',
                    'source_claim_id':claim['id'], 'source_document_id':doc_id, 'rationale_indices':indices})
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/scifact/data'))
    parser.add_argument('--output',type=Path,default=Path('evaluation/benchmarks/scifact_support_v1.json'))
    args=parser.parse_args(argv)
    if args.output.exists(): raise FileExistsError('Use a new benchmark version')
    def read(name): return [json.loads(line) for line in (args.data_dir/name).read_text().splitlines() if line.strip()]
    corpus={str(doc['doc_id']):doc for doc in read('corpus.jsonl')}
    dev=convert_claims(read('claims_train.jsonl'),corpus,split='dev',source_split='train')
    test=convert_claims(read('claims_dev.jsonl'),corpus,split='test',source_split='dev')
    if {r['source_claim_id'] for r in dev} & {r['source_claim_id'] for r in test}:
        raise ValueError('Claim identity overlaps across source partitions')
    test_contents = {(r['claim'].casefold().strip(),r['passage_id']) for r in test}
    before = len(dev)
    dev = [r for r in dev if (r['claim'].casefold().strip(),r['passage_id']) not in test_contents]
    benchmark={'excluded_dev_content_overlaps': before-len(dev), 'benchmark_version':'scifact-rationales-1.0','status':'frozen','fixture_only':False,
        'items':dev+test, 'source':SOURCE,
        'annotation_provenance':{'annotation_source':'SciFact published expert annotations',
                                'human_ground_truth':True,'independent_human_review':False,
                                'note':'Labels imported from the published dataset; no new project-team human review.'},
        'licenses':{'claims_and_labels':'CC BY 4.0','abstracts':'ODC-By 1.0','source':SOURCE+'/blob/master/LICENSE.md'},
        'task_boundary':'Claim/evidence rationale classification only; biomedical distribution; no PARTIALLY_SUPPORTED or UNSUPPORTED labels; not a paper-pair or retrieval benchmark.',
        'partitions':{'dev':'official train','test':'official dev; official test labels unavailable'},
        'source_sha256':{name:hashlib.sha256((args.data_dir/name).read_bytes()).hexdigest() for name in ['corpus.jsonl','claims_train.jsonl','claims_dev.jsonl']},
        'class_counts':{s:dict(Counter(r['label'] for r in rows)) for s,rows in [('dev',dev),('test',test)]}}
    from src.evaluation.reasoning_benchmark_io import validate_claim_support_benchmark
    validate_claim_support_benchmark(benchmark)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(benchmark,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(benchmark['class_counts'],indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())
