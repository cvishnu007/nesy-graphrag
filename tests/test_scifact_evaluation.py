import json

from src.evaluation.scifact_evaluation import evaluate


def test_external_evaluation_freezes_protocol_and_hides_labels(tmp_path):
    benchmark = tmp_path/'benchmark.json'
    benchmark.write_text(json.dumps({'items': [
        {'item_id':'a','split':'test','claim':'Claim','passage_text':'Evidence','label':'CONTRADICTED'},
        {'item_id':'b','split':'dev','claim':'Other','passage_text':'Other','label':'SUPPORTED'}]}))
    output = tmp_path/'run'
    def predict(items, path, **kwargs):
        assert (output/'protocol.json').exists()
        assert len(items)==1 and 'label' not in items[0]
        assert kwargs['model_revision']
        return [{'item_id':'a','prediction':'CONTRADICTED'}]
    report=evaluate(benchmark,output,predictor=predict)
    assert report['count']==1 and report['accuracy']==1
    row=json.loads((output/'predictions.jsonl').read_text())
    assert row['reference_source']=='published SciFact expert annotations'
