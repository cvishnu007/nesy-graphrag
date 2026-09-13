import json
import pytest
from src.evaluation.refresh_retrieval import grade_candidates, cached_retrievers


def test_judge_is_blinded_and_resumes_only_identical_inputs(tmp_path):
    rows=[{'query_id':'Q1','query':'graphs','split':'dev','paper_id':'p1','title':'Graph paper','abstract':'Evidence.'}]
    class Judge:
        def __init__(self): self.calls=0
        def complete(self,prompt,**kwargs):
            self.calls+=1
            assert '"method"' not in prompt
            return json.dumps({'judgments':[{'paper_id':'p1','grade':2,'notes':'Directly addresses graphs.'}]})
    judge=Judge(); path=tmp_path/'grades.json'
    assert grade_candidates(rows,judge,path,batch_size=2)['Q1/p1']['grade']==2
    grade_candidates(rows,judge,path,batch_size=2)
    assert judge.calls==1
    with pytest.raises(ValueError,match='changed'):
        grade_candidates([{**rows[0],'abstract':'Changed evidence'}],judge,path)


def test_judge_rejects_unknown_ids_and_boolean_grades(tmp_path):
    class Judge:
        def complete(self,*a,**k):return '{"judgments":[{"paper_id":"p1","grade":true,"notes":"bad"}]}'
    with pytest.raises(ValueError):
        grade_candidates([{'query_id':'Q1','query':'q','paper_id':'p1','title':'t','abstract':'a'}],Judge(),tmp_path/'g.json')


def test_cached_retrievers_preserve_original_rank_order():
    rows=[{'query':'q','query_id':'Q1','paper_id':p,'title':p,'abstract':'a'} for p in ['a','b']]
    audit=[{'query_id':'Q1','paper_id':'a','ranks':{'vector':2}}, {'query_id':'Q1','paper_id':'b','ranks':{'vector':1}}]
    assert [r['id'] for r in cached_retrievers(rows,audit)['vector']('q',20)]==['b','a']


def test_frozen_pool_detects_ranking_mutation(tmp_path):
    from src.evaluation.refresh_retrieval import freeze_pool, verify_pool
    for name in ['protocol.json', 'queries.json', 'candidates.csv', 'rankings_audit.jsonl']:
        (tmp_path/name).write_text('original')
    freeze_pool(tmp_path)
    verify_pool(tmp_path)
    (tmp_path/'rankings_audit.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        verify_pool(tmp_path)


def test_runtime_protocol_records_actual_evaluation_and_production_settings():
    from src.evaluation.refresh_retrieval import runtime_retrieval_settings
    from src.evaluation import config
    settings=runtime_retrieval_settings()
    assert settings['evaluation_hybrid_weights']==[config.HYBRID_VECTOR_WEIGHT,config.HYBRID_GRAPH_WEIGHT]
    assert settings['production']['hop_depth']>0


def test_compact_schema_maps_ordered_grades_to_exact_candidate_ids(tmp_path):
    rows=[{'query_id':'Q1','query':'q','paper_id':p,'title':'t','abstract':'a'} for p in ['a','b']]
    class Judge:
        def complete(self,prompt,**kwargs):
            assert kwargs['json_schema']['properties']['grades']['minItems']==2
            return '{"grades":[2,0]}'
    grades=grade_candidates(rows,Judge(),tmp_path/'compact.json',structured=True)
    assert grades['Q1/a']['grade']==2 and grades['Q1/b']['grade']==0
