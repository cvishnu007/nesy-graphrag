import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from groq import Groq
from src.utils.config import GROQ_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from src.storage.neo4j_store import get_driver
from src.pipeline.review import llm_review
from src.pipeline.contradiction import llm_contradict
from src.pipeline.hypothesis import llm_hypothesis

# ── Shared clients — init once ─────────────────────────
_driver      = None
_groq_client = None


def get_groq():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def get_neo4j():
    global _driver
    if _driver is None:
        _driver = get_driver()
    return _driver


def graphrag_query(query: str, mode: str = "review", top_k: int = 10) -> dict:
    """
    Master query function.
    mode: "review" | "contradict" | "hypothesis"
    """
    driver      = get_neo4j()
    groq_client = get_groq()

    if mode == "review":
        return llm_review(groq_client, driver, query, top_k=top_k)
    elif mode == "contradict":
        return llm_contradict(groq_client, driver, query, top_k=top_k)
    elif mode == "hypothesis":
        return llm_hypothesis(groq_client, driver, query, top_k=top_k)
    else:
        raise ValueError(f"mode must be: review / contradict / hypothesis. Got: {mode}")


# ── Quick test when run directly ──────────────────────
if __name__ == "__main__":
    test_query = "graph neural networks for node classification"

    print("\n" + "█"*60)
    print("TEST 1 — LITERATURE REVIEW")
    print("█"*60)
    r1 = graphrag_query(test_query, mode="review")

    print("\n" + "█"*60)
    print("TEST 2 — CONTRADICTION DETECTION")
    print("█"*60)
    r2 = graphrag_query(test_query, mode="contradict")

    print("\n" + "█"*60)
    print("TEST 3 — HYPOTHESIS GENERATION")
    print("█"*60)
    r3 = graphrag_query(test_query, mode="hypothesis")

    print("\n\nFull NeSy-GraphRAG + LLM Pipeline Complete!")
