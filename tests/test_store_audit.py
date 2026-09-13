from src.evaluation.store_audit import compare_stores


def test_id_parity_alone_does_not_hide_stale_text_or_unfinished_ner():
    clean=[{'id':'a','clean_title':'Title','clean_abstract':'Current evidence.'}]
    ner=[{**clean[0],'entities':['evidence']}]
    stored=[{'id':'a','title':'Title','abstract':'Current evidence.'}]
    assert compare_stores(clean,ner,iter(stored),iter(stored))['consistent']
    report=compare_stores(clean,[{**ner[0],'entities':None}],iter([{**stored[0],'abstract':'Old evidence.'}]),iter(stored))
    assert not report['consistent']
    assert report['ner']['unfinished_count']==1
    assert report['chroma']['text_mismatch_count']==1
