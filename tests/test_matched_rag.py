import pytest
from src.evaluation.matched_rag import bounded_passages, compare_reviews


PAPERS = [{'id': 'a', 'title': 'Graph learning', 'abstract': 'Graphs encode relationships. Learning predicts links.', 'year': 2021}]


def test_context_budget_keeps_whole_passages_and_rejects_unknown_evidence():
    from src.pipeline.provenance import format_passage_context
    corpus = {p['id']: p for p in PAPERS}
    passages = bounded_passages(PAPERS + [{'id': 'invented'}], corpus, max_chars=160)
    assert passages
    assert len(format_passage_context(passages)) <= 160
    assert all(p['paper_id'] == 'a' for p in passages)
    assert bounded_passages(PAPERS, corpus, max_chars=1) == []


def test_matched_generation_uses_same_prompt_budget_and_preserves_fabrication_audit():
    from src.pipeline.provenance import passage_id
    class Client:
        def __init__(self): self.calls = []
        def complete(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return f'CLAIM: Graphs encode relationships.\nEVIDENCE: {passage_id("a", 1)}\nCLAIM: Unsupported.\nEVIDENCE: invented'
    client = Client()
    retrievers = {name: lambda q, k: PAPERS for name in ['vector_rag', 'graph_rag']}
    rows = compare_reviews('graph', retrievers, {p['id']: p for p in PAPERS}, client, top_k=2, context_chars=1000, max_tokens=100)
    assert len(rows) == 2
    assert client.calls[0] == client.calls[1]
    for row in rows:
        assert row['provenance']['stats']['grounded_claims'] == 1
        assert row['provenance']['stats']['unsupported_claims'] == 1
        assert row['context_chars'] <= 1000
        assert row['raw_output']


def test_invalid_budget_is_rejected_before_retrieval():
    with pytest.raises(ValueError):
        compare_reviews('q', {}, {}, None, context_chars=0)
