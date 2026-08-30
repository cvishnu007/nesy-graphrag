"""Configuration used only by retrieval evaluation experiments."""

import os

from dotenv import load_dotenv


load_dotenv()


BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))
GRAPH_ONLY_CANDIDATE_LIMIT = int(
    os.getenv("GRAPH_ONLY_CANDIDATE_LIMIT", "100")
)

if BM25_K1 <= 0:
    raise ValueError("BM25_K1 must be greater than zero")
if not 0 <= BM25_B <= 1:
    raise ValueError("BM25_B must be between zero and one")
if GRAPH_ONLY_CANDIDATE_LIMIT <= 0:
    raise ValueError("GRAPH_ONLY_CANDIDATE_LIMIT must be greater than zero")
