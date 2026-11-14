import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PLACEHOLDER_VALUES = {"replace_me", "your_key", "your_api_key", "none"}


def is_configured(value, *placeholder_markers):
    if value is None:
        return False

    text = str(value).strip()
    if not text:
        return False

    lowered = text.lower()
    if lowered in PLACEHOLDER_VALUES:
        return False

    return not any(marker.lower() in lowered for marker in placeholder_markers)


# ── Source selection ───────────────────────────────────
DATA_SOURCE = os.getenv("DATA_SOURCE", "s2").strip().lower()
if DATA_SOURCE not in {"arxiv", "s2"}:
    raise ValueError("DATA_SOURCE must be 'arxiv' or 's2'")

# ── Neo4j ─────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_ALLOW_RESET = os.getenv("NEO4J_ALLOW_RESET", "false").strip().lower() in {
    "1", "true", "yes", "on"
}

# ── Groq ──────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

# ── Semantic Scholar ──────────────────────────────────
SEMANTIC_SCHOLAR_API_KEY  = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
SEMANTIC_SCHOLAR_BASE_URL = os.getenv(
    "SEMANTIC_SCHOLAR_BASE_URL",
    "https://api.semanticscholar.org/graph/v1"
)
SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC = float(
    os.getenv("SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC", "1.05")
)
SEMANTIC_SCHOLAR_TIMEOUT_SEC = int(os.getenv("SEMANTIC_SCHOLAR_TIMEOUT_SEC", "45"))
SEMANTIC_SCHOLAR_MAX_RETRIES = int(os.getenv("SEMANTIC_SCHOLAR_MAX_RETRIES", "6"))
S2_QUERY = os.getenv("S2_QUERY", "computer science")
S2_LIMIT = int(os.getenv("S2_LIMIT", "10000"))
S2_QUERIES = tuple(
    dict.fromkeys(
        query.strip()
        for query in os.getenv("S2_QUERIES", S2_QUERY).split(";")
        if query.strip()
    )
)
if not S2_QUERIES:
    raise ValueError("S2_QUERIES must contain at least one query")
S2_LIMIT_PER_QUERY = int(os.getenv("S2_LIMIT_PER_QUERY", str(S2_LIMIT)))
if S2_LIMIT_PER_QUERY <= 0:
    raise ValueError("S2_LIMIT_PER_QUERY must be greater than zero")
S2_INCLUDE_EXISTING = os.getenv("S2_INCLUDE_EXISTING", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
S2_PAGE_SIZE = int(os.getenv("S2_PAGE_SIZE", "1000"))
S2_YEAR = os.getenv("S2_YEAR", "2020-2025")
S2_FIELDS_OF_STUDY = os.getenv("S2_FIELDS_OF_STUDY", "Computer Science")
S2_PUBLICATION_TYPES = os.getenv("S2_PUBLICATION_TYPES", "")
S2_SORT = os.getenv("S2_SORT", "citationCount:desc")
S2_BATCH_SIZE = int(os.getenv("S2_BATCH_SIZE", "500"))
S2_MAX_REFS_PER_PAPER = int(os.getenv("S2_MAX_REFS_PER_PAPER", "200"))
S2_CHECKPOINT_FILE = os.getenv(
    "S2_CHECKPOINT_FILE", "./data/s2_ingestion_checkpoint.json"
)

# ── Data paths ────────────────────────────────────────
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chromadb")

ARXIV_RAW_FILE   = os.getenv("ARXIV_RAW_FILE",   "./data/arxiv_raw.json")
ARXIV_CLEAN_FILE = os.getenv("ARXIV_CLEAN_FILE", "./data/arxiv_clean.json")
ARXIV_NER_FILE   = os.getenv("ARXIV_NER_FILE",   "./data/arxiv_ner.json")

S2_RAW_FILE   = os.getenv("S2_RAW_FILE",   "./data/s2_raw.json")
S2_CLEAN_FILE = os.getenv("S2_CLEAN_FILE", "./data/s2_clean.json")
S2_NER_FILE   = os.getenv("S2_NER_FILE",   "./data/s2_ner.json")

RAW_FILE = os.getenv(
    "RAW_FILE",
    S2_RAW_FILE if DATA_SOURCE == "s2" else ARXIV_RAW_FILE
)
CLEAN_FILE = os.getenv(
    "CLEAN_FILE",
    S2_CLEAN_FILE if DATA_SOURCE == "s2" else ARXIV_CLEAN_FILE
)
NER_FILE = os.getenv(
    "NER_FILE",
    S2_NER_FILE if DATA_SOURCE == "s2" else ARXIV_NER_FILE
)
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", f"{DATA_SOURCE}_papers")
USE_REAL_CITATIONS = os.getenv(
    "USE_REAL_CITATIONS",
    "true" if DATA_SOURCE == "s2" else "false"
).strip().lower() in {"1", "true", "yes", "on"}

# ── Models ────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "allenai-specter")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
LLM_MODEL          = os.getenv("LLM_MODEL",          "openai/gpt-oss-120b")
LLM_MODEL_FALLBACK = os.getenv("LLM_MODEL_FALLBACK", "llama-3.1-8b-instant")
GROQ_MAX_RETRIES   = int(os.getenv("GROQ_MAX_RETRIES", 4))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
if LLM_PROVIDER not in {"ollama", "groq"}:
    raise ValueError("LLM_PROVIDER must be 'ollama' or 'groq'")


def _positive_llm_int(name, default):
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


LLM_MAX_RETRIES = _positive_llm_int("LLM_MAX_RETRIES", GROQ_MAX_RETRIES)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct").strip()
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "").strip()
OLLAMA_TIMEOUT_SEC = _positive_llm_int("OLLAMA_TIMEOUT_SEC", 300)
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m").strip()
OLLAMA_CONTEXT_LENGTH = _positive_llm_int("OLLAMA_CONTEXT_LENGTH", 8192)

for _name, _value in {
    "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
    "OLLAMA_MODEL": OLLAMA_MODEL,
    "OLLAMA_KEEP_ALIVE": OLLAMA_KEEP_ALIVE,
}.items():
    if not _value:
        raise ValueError(f"{_name} must be non-empty")

# ── Pipeline settings ─────────────────────────────────
BATCH_SIZE          = int(os.getenv("BATCH_SIZE",          64))
NEO4J_BATCH_SIZE    = int(os.getenv("NEO4J_BATCH_SIZE",    500))
TOP_K               = int(os.getenv("TOP_K",               10))
HOP_DEPTH           = int(os.getenv("HOP_DEPTH",           2))
RRF_K               = int(os.getenv("RRF_K",               60))
NEURAL_FUSION_WEIGHT = float(os.getenv("NEURAL_FUSION_WEIGHT", "1.0"))
GRAPH_FUSION_WEIGHT  = float(os.getenv("GRAPH_FUSION_WEIGHT",  "1.0"))
GRAPH_HIGH_SEMANTIC_THRESHOLD = float(
    os.getenv("GRAPH_HIGH_SEMANTIC_THRESHOLD", "0.85")
)
GRAPH_SEMANTIC_FLOOR = float(os.getenv("GRAPH_SEMANTIC_FLOOR", "0.75"))
GRAPH_MIN_QUERY_TERM_COVERAGE = float(
    os.getenv("GRAPH_MIN_QUERY_TERM_COVERAGE", "0.75")
)
GRAPH_STRONG_CONNECTIONS = int(os.getenv("GRAPH_STRONG_CONNECTIONS", "10"))

# -- Experimental retrieval options (defaults reproduce the frozen protocol) --
# Graph gate: "absolute" uses the fixed thresholds above; "relative" keeps a
# graph neighbour whose similarity is within GRAPH_RELATIVE_MARGIN of the
# weakest neural seed, so the gate adapts to each query's similarity band.
GRAPH_GATE_MODE = os.getenv("GRAPH_GATE_MODE", "absolute").strip().lower()
if GRAPH_GATE_MODE not in {"absolute", "relative"}:
    raise ValueError("GRAPH_GATE_MODE must be 'absolute' or 'relative'")
GRAPH_RELATIVE_MARGIN = float(os.getenv("GRAPH_RELATIVE_MARGIN", "0.03"))
GRAPH_CANDIDATE_LIMIT = int(os.getenv("GRAPH_CANDIDATE_LIMIT", "20"))
if GRAPH_CANDIDATE_LIMIT < 1:
    raise ValueError("GRAPH_CANDIDATE_LIMIT must be at least 1")
# Relationship types traversed during expansion, separated by "|".
GRAPH_EXPANSION_EDGES = tuple(
    edge.strip().upper()
    for edge in os.getenv("GRAPH_EXPANSION_EDGES", "CITES").split("|")
    if edge.strip()
)
# CITES_ADDED and COUPLED_WITH come from src/ingestion/citation_densify.py (opt-in; defaults unchanged)
ALLOWED_EXPANSION_EDGES = {"CITES", "SIMILAR_TO", "CITES_ADDED", "COUPLED_WITH"}
if not GRAPH_EXPANSION_EDGES or not set(GRAPH_EXPANSION_EDGES) <= ALLOWED_EXPANSION_EDGES:
    raise ValueError(f"GRAPH_EXPANSION_EDGES must use only {sorted(ALLOWED_EXPANSION_EDGES)}")
# Hide a result whose abstract is near-identical (cosine >= 0.995) to a higher-ranked
# result. Pairs come from evaluation/near_duplicate_pairs.json; no data is deleted.
COLLAPSE_NEAR_DUPLICATES = os.getenv("COLLAPSE_NEAR_DUPLICATES", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
NEAR_DUPLICATE_FILE = os.getenv(
    "NEAR_DUPLICATE_FILE",
    str(Path(__file__).resolve().parents[2] / "evaluation/near_duplicate_pairs.json"),
)
# Lexical third arm for reciprocal-rank fusion; 0 disables it.
BM25_FUSION_WEIGHT = float(os.getenv("BM25_FUSION_WEIGHT", "0.0"))
if BM25_FUSION_WEIGHT < 0:
    raise ValueError("BM25_FUSION_WEIGHT cannot be negative")
# Text embedded into Chroma: "abstract" (historical index) or "title_abstract"
# (SPECTER's training format: original-case title [SEP] abstract).
EMBEDDING_TEXT_MODE = os.getenv("EMBEDDING_TEXT_MODE", "abstract").strip().lower()
if EMBEDDING_TEXT_MODE not in {"abstract", "title_abstract", "title_abstract_plain"}:
    raise ValueError("EMBEDDING_TEXT_MODE must be abstract, title_abstract, or title_abstract_plain")
# Instruction prepended to queries only (e.g. BGE models). Documents are never prefixed.
EMBEDDING_QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX", "")
# Concepts flagged generic by src.storage.graph_enrichment are ignored by
# concept-based reasoning when this is on. Off by default: with it on, the
# concept-overlap contradiction generator finds no candidate pairs on the
# current graph (measured November 13, 2026), because its pairs came from
# generic hub phrases.
# Concept graph for reasoning: "spacy" (historical noun chunks) or "scientific"
# (SciERC-trained entities; build with graph_enrichment scientific-entities).
CONCEPT_SOURCE = os.getenv("CONCEPT_SOURCE", "spacy").strip().lower()
if CONCEPT_SOURCE not in {"spacy", "scientific"}:
    raise ValueError("CONCEPT_SOURCE must be 'spacy' or 'scientific'")
CONCEPT_EXCLUDE_GENERIC = os.getenv("CONCEPT_EXCLUDE_GENERIC", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
CITES_THRESHOLD     = int(os.getenv("CITES_THRESHOLD",     2))
MAX_AUTHORS         = int(os.getenv("MAX_AUTHORS",         5))
MAX_CONCEPTS        = int(os.getenv("MAX_CONCEPTS",        10))
MIN_ABSTRACT_WORDS  = int(os.getenv("MIN_ABSTRACT_WORDS",  30))

for _name, _value in {
    "GRAPH_HIGH_SEMANTIC_THRESHOLD": GRAPH_HIGH_SEMANTIC_THRESHOLD,
    "GRAPH_SEMANTIC_FLOOR": GRAPH_SEMANTIC_FLOOR,
    "GRAPH_MIN_QUERY_TERM_COVERAGE": GRAPH_MIN_QUERY_TERM_COVERAGE,
}.items():
    if not 0.0 <= _value <= 1.0:
        raise ValueError(f"{_name} must be between 0 and 1")
if GRAPH_SEMANTIC_FLOOR > GRAPH_HIGH_SEMANTIC_THRESHOLD:
    raise ValueError(
        "GRAPH_SEMANTIC_FLOOR must not exceed GRAPH_HIGH_SEMANTIC_THRESHOLD"
    )
if GRAPH_STRONG_CONNECTIONS < 1:
    raise ValueError("GRAPH_STRONG_CONNECTIONS must be at least 1")
EVALUATION_START_YEAR = int(os.getenv("EVALUATION_START_YEAR", "2020"))
EVALUATION_END_YEAR = int(
    os.getenv("EVALUATION_END_YEAR", "2025" if DATA_SOURCE == "s2" else "2024")
)
if EVALUATION_END_YEAR < EVALUATION_START_YEAR:
    raise ValueError("EVALUATION_END_YEAR must be greater than or equal to EVALUATION_START_YEAR")
CONTRADICTION_MIN_SHARED_CONCEPTS = int(
    os.getenv("CONTRADICTION_MIN_SHARED_CONCEPTS", "2")
)
CONTRADICTION_MIN_CONCEPT_JACCARD = float(
    os.getenv("CONTRADICTION_MIN_CONCEPT_JACCARD", "0.10")
)
CONTRADICTION_CANDIDATE_POOL = int(
    os.getenv("CONTRADICTION_CANDIDATE_POOL", "25")
)
CONTRADICTION_MIN_CONFIDENCE = float(
    os.getenv("CONTRADICTION_MIN_CONFIDENCE", "0.70")
)
# Candidate generator: "concept" (shared noun-phrase overlap, historical) or
# "nli" (sentence-level NLI contradiction scores; see pipeline/contradiction_nli.py).
CONTRADICTION_CANDIDATE_MODE = os.getenv("CONTRADICTION_CANDIDATE_MODE", "concept").strip().lower()
if CONTRADICTION_CANDIDATE_MODE not in {"concept", "nli", "citation"}:
    raise ValueError("CONTRADICTION_CANDIDATE_MODE must be 'concept', 'nli' or 'citation'")
# Corpus-wide citation-grounded index (src/evaluation/contradiction_index.py), used when mode is "citation".
CONTRADICTION_INDEX_FILE = os.getenv(
    "CONTRADICTION_INDEX_FILE", "results/reasoning/contradiction_index_20261114/index.jsonl")
CONTRADICTION_INDEX_RETRIEVE_K = int(os.getenv("CONTRADICTION_INDEX_RETRIEVE_K", "50"))
CONTRADICTION_YEAR_WEIGHT = float(os.getenv("CONTRADICTION_YEAR_WEIGHT", "0.25"))
if not 0 <= CONTRADICTION_YEAR_WEIGHT <= 1:
    raise ValueError("CONTRADICTION_YEAR_WEIGHT must be between 0 and 1")
CONTRADICTION_REQUIRE_DIFFERENT_YEAR = os.getenv(
    "CONTRADICTION_REQUIRE_DIFFERENT_YEAR", "true"
).strip().lower() in {"1", "true", "yes", "on"}
# Schema-constrained JSON verdicts (Ollama only); the legacy line format is
# still parsed, so stored analyses remain readable.
CONTRADICTION_STRUCTURED_OUTPUT = os.getenv(
    "CONTRADICTION_STRUCTURED_OUTPUT", "true"
).strip().lower() in {"1", "true", "yes", "on"}
HYPOTHESIS_MIN_SHARED_CONCEPTS = int(
    os.getenv("HYPOTHESIS_MIN_SHARED_CONCEPTS", "2")
)
HYPOTHESIS_MIN_QUERY_SUPPORT = int(
    os.getenv("HYPOTHESIS_MIN_QUERY_SUPPORT", "2")
)
HYPOTHESIS_CANDIDATE_POOL = int(
    os.getenv("HYPOTHESIS_CANDIDATE_POOL", "25")
)
# Relationships that disqualify a "structural hole": a candidate linked to any
# query paper by one of these edge types is already connected.
# "structural_hole" (original; rated below a vector baseline in round 1) or "bridge"
# (src/pipeline/hypothesis_bridge.py; dev-selected, time-sliced signals validated, round 2 pending).
HYPOTHESIS_GENERATOR = os.getenv("HYPOTHESIS_GENERATOR", "structural_hole").strip().lower()
if HYPOTHESIS_GENERATOR not in {"structural_hole", "bridge"}:
    raise ValueError("HYPOTHESIS_GENERATOR must be 'structural_hole' or 'bridge'")
HYPOTHESIS_HOLE_EDGES = tuple(
    edge.strip().upper()
    for edge in os.getenv("HYPOTHESIS_HOLE_EDGES", "CITES").split("|")
    if edge.strip()
)
if not HYPOTHESIS_HOLE_EDGES or not set(HYPOTHESIS_HOLE_EDGES) <= ALLOWED_EXPANSION_EDGES:
    raise ValueError(f"HYPOTHESIS_HOLE_EDGES must use only {sorted(ALLOWED_EXPANSION_EDGES)}")

# -- Reasoning evaluation -------------------------------------------------
CONTRADICTION_BENCHMARK_FILE = os.getenv(
    "CONTRADICTION_BENCHMARK_FILE",
    str(Path(__file__).resolve().parents[2] / "teammate2/benchmarks/contradiction_pairs.json"),
)
CLAIM_SUPPORT_BENCHMARK_FILE = os.getenv(
    "CLAIM_SUPPORT_BENCHMARK_FILE",
    str(Path(__file__).resolve().parents[2] / "teammate2/benchmarks/claim_support.json"),
)
HYPOTHESIS_BENCHMARK_FILE = os.getenv(
    "HYPOTHESIS_BENCHMARK_FILE",
    str(Path(__file__).resolve().parents[2] / "teammate2/benchmarks/hypothesis_ratings.json"),
)
REASONING_RESULTS_DIR = os.getenv("REASONING_RESULTS_DIR", "results/reasoning")
SEMANTIC_SUPPORT_MODEL = os.getenv("SEMANTIC_SUPPORT_MODEL", "unconfigured")
SEMANTIC_SUPPORT_MIN_CONFIDENCE = float(
    os.getenv("SEMANTIC_SUPPORT_MIN_CONFIDENCE", "0.70")
)

for _name, _value in {
    "CONTRADICTION_BENCHMARK_FILE": CONTRADICTION_BENCHMARK_FILE,
    "CLAIM_SUPPORT_BENCHMARK_FILE": CLAIM_SUPPORT_BENCHMARK_FILE,
    "HYPOTHESIS_BENCHMARK_FILE": HYPOTHESIS_BENCHMARK_FILE,
    "REASONING_RESULTS_DIR": REASONING_RESULTS_DIR,
    "SEMANTIC_SUPPORT_MODEL": SEMANTIC_SUPPORT_MODEL,
}.items():
    if not str(_value).strip():
        raise ValueError(f"{_name} must be non-empty")
if not 0 <= SEMANTIC_SUPPORT_MIN_CONFIDENCE <= 1:
    raise ValueError("SEMANTIC_SUPPORT_MIN_CONFIDENCE must be between zero and one")
