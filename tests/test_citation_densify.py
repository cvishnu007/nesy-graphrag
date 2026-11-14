from src.ingestion.citation_densify import abstract_from_index, build_plan, coupling_edges


def test_coupling_links_papers_sharing_references():
    refs = {"a": {"x", "y", "z"}, "b": {"x", "y"}, "c": {"q"}}
    edges = coupling_edges(refs, top_k=5)
    assert [(e["source"], e["target"], e["shared"]) for e in edges] == [("a", "b", 2)]


def test_build_plan_merges_sources_without_duplicating_original_cites():
    corpus = [{"id": "a", "references": "['b', 'ext1']"}, {"id": "b", "references": "[]"},
              {"id": "c", "references": "[]"}]
    supplement = {"c": ["b", "ext1", "ext2"]}
    openalex = {"a": {"openalex_id": "W1", "refs": ["W2", "W9", "W8"]},
                "b": {"openalex_id": "W2", "refs": ["W9", "W8"]},
                "c": {"openalex_id": "W3", "refs": ["W9"]}}
    plan = build_plan(corpus, supplement, openalex, min_citing=2)
    assert ("a", "b") not in plan["cites_added"]          # already an original CITES edge
    assert ("c", "b") in plan["cites_added"]              # from the S2 supplement
    assert {"W9", "W8"} == set(plan["closure"])           # external, cited by >= 2 corpus papers
    assert any(e["source"] == "a" and e["target"] == "b" for e in plan["coupled_with"])
    assert plan["papers_with_references_before"] == 1 and plan["papers_with_references"] == 3


def test_abstract_rebuilt_from_inverted_index():
    assert abstract_from_index({"graphs": [1], "learning": [0, 2]}) == "learning graphs learning"
