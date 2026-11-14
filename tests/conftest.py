import os

import pytest

# Unit tests must not depend on a developer's local .env. Pin every selectable
# retrieval/reasoning option to its documented default before src.* modules
# import src.utils.config (python-dotenv never overrides existing variables).
_TEST_DEFAULTS = {
    "EMBEDDING_QUERY_PREFIX": "",
    "EMBEDDING_TEXT_MODE": "abstract",
    "GRAPH_GATE_MODE": "absolute",
    "GRAPH_EXPANSION_EDGES": "CITES",
    "GRAPH_CANDIDATE_LIMIT": "20",
    "BM25_FUSION_WEIGHT": "0.0",
    "CONCEPT_SOURCE": "spacy",
    "CONCEPT_EXCLUDE_GENERIC": "false",
    "CONTRADICTION_CANDIDATE_MODE": "concept",
    "HYPOTHESIS_HOLE_EDGES": "CITES",
    "HYPOTHESIS_GENERATOR": "structural_hole",
}
os.environ.update(_TEST_DEFAULTS)


@pytest.fixture
def paper_factory():
    def make_paper(paper_id, score=0.8):
        return {
            "id": paper_id,
            "title": paper_id,
            "abstract": "",
            "year": 2024,
            "category": "Computer Science",
            "score": score,
        }

    return make_paper
