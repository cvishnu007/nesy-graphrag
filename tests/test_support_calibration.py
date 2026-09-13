import pytest
from src.evaluation.support_calibration import select_partial_floor, remap_predictions, coverage_report


def test_selection_uses_only_development_records():
    rows = [{'item_id': 'a', 'split': 'dev', 'label': 'PARTIALLY_SUPPORTED',
             'nli_scores': {'entailment': .1, 'neutral': .85, 'contradiction': .05}}]
    selected = select_partial_floor(rows, floors=(.2, .05))
    assert selected['partial_entailment_floor'] == .05
    assert selected['selection_split'] == 'dev'
    with pytest.raises(ValueError, match='dev'):
        select_partial_floor([{**rows[0], 'split': 'test'}])


def test_prediction_remapping_cannot_access_reference_labels():
    row = {'item_id': 'a', 'split': 'test', 'label': 'CONTRADICTED',
           'nli_scores': {'entailment': .1, 'neutral': .85, 'contradiction': .05}}
    assert remap_predictions([row], floor=.05)[0]['prediction'] == 'PARTIALLY_SUPPORTED'
    assert 'label' not in remap_predictions([row], floor=.05)[0]
    with pytest.raises(ValueError):
        remap_predictions([{**row, 'nli_scores': {'entailment': float('nan'), 'neutral': 1., 'contradiction': 0.}}], floor=.2)


def test_coverage_exposes_missing_positive_classes():
    report = coverage_report([{'split': 'dev', 'label': 'AGREEMENT'}, {'split': 'test', 'label': 'CONTRADICTION'}])
    assert report['ready_for_class_comparison'] is False
    assert report['counts']['test']['CONTRADICTION'] == 1
    assert report['additional_required']['dev']['CONTRADICTION'] == 10
