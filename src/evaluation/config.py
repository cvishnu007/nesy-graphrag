"""Configuration used only by retrieval evaluation experiments."""

import os

from dotenv import load_dotenv


load_dotenv()


HYBRID_VECTOR_WEIGHT = float(os.getenv("EVALUATION_HYBRID_VECTOR_WEIGHT", "16.0"))
HYBRID_GRAPH_WEIGHT = float(os.getenv("EVALUATION_HYBRID_GRAPH_WEIGHT", "1.0"))
GRAPH_ONLY_CANDIDATE_LIMIT = int(
    os.getenv("GRAPH_ONLY_CANDIDATE_LIMIT", "100")
)

if GRAPH_ONLY_CANDIDATE_LIMIT <= 0:
    raise ValueError("GRAPH_ONLY_CANDIDATE_LIMIT must be greater than zero")
if min(HYBRID_VECTOR_WEIGHT, HYBRID_GRAPH_WEIGHT) < 0:
    raise ValueError("Hybrid retrieval weights cannot be negative")
if HYBRID_VECTOR_WEIGHT + HYBRID_GRAPH_WEIGHT <= 0:
    raise ValueError("At least one hybrid retrieval weight must be positive")
