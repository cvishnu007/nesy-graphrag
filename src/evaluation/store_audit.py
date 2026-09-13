"""Read-only identity and evidence-text checks for the local demo's four stores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def compare_stores(clean, ner, vectors, graph):
    canonical = {p['id']: (p['clean_title'], p['clean_abstract']) for p in clean}
    if not canonical or len(canonical) != len(clean):
        raise ValueError('Clean corpus must have unique, non-empty paper IDs')
    expected = set(canonical)
    report = {'paper_count': len(expected), 'paper_ids_sha256': hashlib.sha256(
        '\n'.join(sorted(expected)).encode()).hexdigest()}
    for name, rows in [('ner',ner),('chroma',vectors),('neo4j',graph)]:
        seen, mismatch, duplicates, unfinished = set(), [], 0, 0
        for row in rows:
            pid = row['id']
            duplicates += pid in seen
            seen.add(pid)
            text = (row.get('clean_title', row.get('title')), row.get('clean_abstract', row.get('abstract')))
            if pid in canonical and text != canonical[pid]:
                mismatch.append(pid)
            if name == 'ner' and not isinstance(row.get('entities'), list):
                unfinished += 1
        report[name] = {'count': len(seen), 'missing_count':len(expected-seen),
            'extra_count':len(seen-expected), 'duplicate_count':duplicates,
            'text_mismatch_count':len(mismatch), 'text_mismatch_sample':mismatch[:10],
            'unfinished_count':unfinished}
    report['consistent'] = all(not report[name][key] for name in ['ner','chroma','neo4j']
        for key in ['missing_count','extra_count','duplicate_count','text_mismatch_count','unfinished_count'])
    return report


def audit():
    from src.storage.chroma_store import get_collection
    from src.storage.neo4j_store import get_driver
    from src.utils.config import CLEAN_FILE, NER_FILE, RAW_FILE, CHROMA_COLLECTION, EMBEDDING_MODEL
    clean_bytes, ner_bytes = Path(CLEAN_FILE).read_bytes(), Path(NER_FILE).read_bytes()
    clean, ner = json.loads(clean_bytes), json.loads(ner_bytes)
    collection = get_collection()
    ids = collection.get(include=[])['ids']
    def vectors():
        for offset in range(0,len(ids),1000):
            batch = collection.get(ids=ids[offset:offset+1000],include=['documents','metadatas'])
            for pid, abstract, metadata in zip(batch['ids'],batch['documents'],batch['metadatas']):
                yield {'id':pid,'title':metadata['title'],'abstract':abstract}
    driver = get_driver()
    try:
        with driver.session() as session:
            graph = session.run('MATCH (p:Paper) RETURN p.id AS id, p.title AS title, p.abstract AS abstract')
            report = compare_stores(clean,ner,vectors(),graph)
            report['graph_nodes'] = {r['label']:r['count'] for r in session.run(
                'MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count')}
            report['graph_relationships'] = {r['type']:r['count'] for r in session.run(
                'MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count')}
    finally:
        driver.close()
    report.update({'clean_sha256':hashlib.sha256(clean_bytes).hexdigest(),
        'ner_sha256':hashlib.sha256(ner_bytes).hexdigest(),
        'raw_sha256':hashlib.sha256(Path(RAW_FILE).read_bytes()).hexdigest(),
        'collection':CHROMA_COLLECTION,'embedding_model':EMBEDDING_MODEL,
        'scope':'Exact IDs and title/abstract equality; does not re-encode or verify individual vector values.'})
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Use a new audit output')
    report=audit()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return int(not report['consistent'])


if __name__=='__main__':
    raise SystemExit(main())
