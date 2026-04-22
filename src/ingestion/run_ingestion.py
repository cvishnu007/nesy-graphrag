import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import DATA_SOURCE


def run():
    if DATA_SOURCE == "s2":
        from src.ingestion import semantic_scholar_fetcher
        print("DATA_SOURCE=s2 → running Semantic Scholar ingestion")
        semantic_scholar_fetcher.run()
        return

    from src.ingestion import arxiv_fetcher
    print("DATA_SOURCE=arxiv → running ArXiv ingestion")
    papers = arxiv_fetcher.fetch_papers()
    arxiv_fetcher.preprocess(papers)


if __name__ == "__main__":
    run()
