from src.evaluation.scifact_benchmark import convert_claims
import pytest


def test_multisentence_rationale_is_kept_together_and_labels_are_external():
    corpus = {'42': {'title': 'Paper', 'abstract': ['First fact.', 'Second fact.']}}
    claims = [{'id': 7, 'claim': 'A composite claim.', 'evidence': {'42': [{'sentences': [0,1], 'label': 'CONTRADICT'}]}}]
    rows = convert_claims(claims, corpus, split='test', source_split='dev')
    assert len(rows) == 1
    assert rows[0]['passage_text'] == 'First fact. Second fact.'
    assert rows[0]['label'] == 'CONTRADICTED'
    assert rows[0]['source_split'] == 'dev'


def test_unlabelled_claims_are_not_invented_as_negative_labels():
    assert convert_claims([{'id': 8, 'claim': 'unknown', 'evidence': {}}], {}, split='dev', source_split='train') == []
    with pytest.raises(ValueError):
        convert_claims([{'id': 7, 'claim': 'x', 'evidence': {'missing': [{'sentences':[0], 'label':'SUPPORT'}]}}], {}, split='dev', source_split='train')


def test_repeated_source_claim_text_is_deduplicated_without_splitting_rationales():
    corpus = {'42': {'title': 'Paper', 'abstract': ['Fact.']}}
    claim = {'id': 7, 'claim': 'A claim.', 'evidence': {'42': [{'sentences':[0], 'label':'SUPPORT'}]}}
    assert len(convert_claims([claim, {**claim, 'id':8}], corpus, split='dev', source_split='train')) == 1
