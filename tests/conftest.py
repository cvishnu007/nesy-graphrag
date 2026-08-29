import pytest


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
