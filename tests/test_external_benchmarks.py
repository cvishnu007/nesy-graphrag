import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.evaluation.external_benchmarks import document_text, load_litsearch, load_scidocs, rrf, score
from src.evaluation.significance import paired_permutation_test


def test_document_text_uses_model_input_format():
    paper = {"title": " A Title ", "abstract": "Body text."}
    assert document_text(paper, "sep", "[SEP]") == "A Title[SEP]Body text."
    assert document_text(paper, "dot") == "A Title. Body text."
    assert document_text({"title": "Only", "abstract": ""}, "dot") == "Only"


def test_rrf_rewards_agreement_and_breaks_ties_by_id():
    fused = rrf({"a": {"q": ["x", "y"]}, "b": {"q": ["y", "z"]}}, {"a": 1.0, "b": 1.0}, depth=3)
    assert fused["q"][0] == "y"
    assert set(fused["q"]) == {"x", "y", "z"}


def test_score_uses_benchmark_labels():
    bench = {"queries": {"q1": "t"}, "qrels": {"q1": {"d1": 1, "d2": 0}}}
    rows, metrics = score(bench, {"q1": ["d2", "d1"]})
    assert rows[0]["recall@5"] == 1.0
    assert metrics["mrr"]["mean"] == pytest.approx(0.5)


def test_permutation_test_detects_consistent_gain_and_null():
    gain = paired_permutation_test([0.1] * 40, samples=2000, bootstrap_samples=2000)
    assert gain["monte_carlo_two_sided_p"] < 0.01
    assert gain["bootstrap_95_ci"][0] > 0
    null = paired_permutation_test([0.1, -0.1] * 20, samples=2000, bootstrap_samples=2000)
    assert null["monte_carlo_two_sided_p"] > 0.5
    with pytest.raises(ValueError):
        paired_permutation_test([])


def test_nli_candidates_rank_by_strongest_contradicting_findings():
    from src.pipeline.contradiction_nli import finding_sentences, rank_contradiction_candidates

    papers = [
        {"id": "p1", "title": "A", "year": 2021,
         "abstract": "We study pruning. Our results show that pruning improves accuracy on ImageNet."},
        {"id": "p2", "title": "B", "year": 2023,
         "abstract": "We revisit pruning. We find that pruning does not improve accuracy on ImageNet."},
        {"id": "p3", "title": "C", "year": 2022,
         "abstract": "We propose a new database index. Experiments show it reduces query latency significantly."},
    ]
    assert [row["sentence_index"] for row in finding_sentences(papers[0])] == [2]

    def embed(texts):  # pruning sentences share a topic; the database one does not
        return [[1.0, 0.0] if "pruning" in text else [0.0, 1.0] for text in texts]

    def scorer(pairs):
        return [0.9 if "does not" in a + b else 0.1 for a, b in pairs]

    ranked = rank_contradiction_candidates(papers, top_k=5, embed=embed, scorer=scorer)
    assert len(ranked) == 1  # off-topic pairs are never scored
    top = ranked[0]
    assert {top["paper1"]["id"], top["paper2"]["id"]} == {"p1", "p2"}
    assert top["nli_contradiction"] == 0.9 and top["candidate_method"] == "nli_sentence_pairs"
    assert "does not improve" in top["evidence_2"]["text"] and top["evidence_1"]["id"].endswith("-S002")


@pytest.mark.parametrize("sentence,expected", [
    ("Graph neural networks (GNNs) have achieved state-of-the-art performance in many fields.", False),
    ("In recent years, deep learning has improved accuracy across many vision tasks.", False),
    ("We show that deep graph attention networks suffer gradient explosion during training.", True),
    ("Our results indicate that pruning does not improve accuracy on ImageNet.", True),
    ("GNNs effectively learn from highly structured data with relationships between nodes.", False),
])
def test_finding_filter_separates_author_results_from_background(sentence, expected):
    from src.pipeline.contradiction_nli import is_finding

    assert is_finding(sentence) is expected


def test_nli_candidates_need_two_papers_with_findings():
    from src.pipeline.contradiction_nli import rank_contradiction_candidates

    papers = [{"id": "p1", "abstract": "We show that x improves y in all settings tested."},
              {"id": "p2", "abstract": "A survey of methods."}]
    assert rank_contradiction_candidates(papers, embed=lambda t: [[1.0]] * len(t), scorer=lambda p: [0.5] * len(p)) == []


def test_contradiction_prompt_includes_focus_sentences():
    from src.pipeline.prompts import build_contradiction_prompt

    p1 = {"title": "A", "year": 2020, "abstract": "x"}
    p2 = {"title": "B", "year": 2021, "abstract": "y"}
    prompt = build_contradiction_prompt(p1, p2, structured=True,
                                        evidence=({"text": "X helps."}, {"text": "X hurts."}))
    assert "FOCUS SENTENCES" in prompt and "X hurts." in prompt
    assert "FOCUS SENTENCES" not in build_contradiction_prompt(p1, p2, structured=True)


def test_loaders_read_benchmark_layout(tmp_path):
    scidocs = tmp_path / "BeIR__scidocs"
    (scidocs / "corpus").mkdir(parents=True)
    (scidocs / "queries").mkdir()
    pq.write_table(pa.table({"_id": ["d1", "d2"], "title": ["T1", "T2"], "text": ["A1", "A2"]}),
                   scidocs / "corpus/corpus-00000-of-00001.parquet")
    pq.write_table(pa.table({"_id": ["q1", "q2"], "text": ["Q1", "Q2"]}), scidocs / "queries/queries-00000-of-00001.parquet")
    (tmp_path / "BeIR__scidocs-qrels").mkdir()
    (tmp_path / "BeIR__scidocs-qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t0\n")
    bench = load_scidocs(tmp_path)
    assert bench["queries"] == {"q1": "Q1"}  # unlabeled q2 dropped
    assert bench["qrels"] == {"q1": {"d1": 1, "d2": 0}}

    lit = tmp_path / "princeton-nlp__LitSearch"
    (lit / "corpus_clean").mkdir(parents=True)
    (lit / "query").mkdir()
    pq.write_table(pa.table({"corpusid": [1, 2], "title": ["P1", "P2"], "abstract": ["a", "b"], "full_paper": ["x", "y"]}),
                   lit / "corpus_clean/full-00000-of-00001.parquet")
    pq.write_table(pa.table({"query_set": ["manual_acl", "inline_acl"], "query": ["find P1", "missing"],
                             "specificity": [1, 0], "quality": [2, 1], "corpusids": [[1], [99]]}),
                   lit / "query/full-00000-of-00001.parquet")
    bench = load_litsearch(tmp_path)
    assert bench["qrels"] == {"LS0000": {"1": 1}}  # query with no target in corpus dropped
    assert bench["query_meta"]["LS0000"]["specificity"] == 1
    assert np.isclose(len(bench["corpus"]), 2)
