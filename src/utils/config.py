import os
from dotenv import load_dotenv

load_dotenv()

# ── Source selection ───────────────────────────────────
DATA_SOURCE = os.getenv("DATA_SOURCE", "arxiv").strip().lower()

# ── Neo4j ─────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

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
S2_PAGE_SIZE = int(os.getenv("S2_PAGE_SIZE", "1000"))
S2_YEAR = os.getenv("S2_YEAR", "2020-2025")
S2_FIELDS_OF_STUDY = os.getenv("S2_FIELDS_OF_STUDY", "Computer Science")
S2_PUBLICATION_TYPES = os.getenv("S2_PUBLICATION_TYPES", "")
S2_SORT = os.getenv("S2_SORT", "citationCount:desc")
S2_BATCH_SIZE = int(os.getenv("S2_BATCH_SIZE", "500"))
S2_MAX_REFS_PER_PAPER = int(os.getenv("S2_MAX_REFS_PER_PAPER", "200"))

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
LLM_MODEL       = os.getenv("LLM_MODEL",       "llama-3.3-70b-versatile")

# ── Pipeline settings ─────────────────────────────────
BATCH_SIZE          = int(os.getenv("BATCH_SIZE",          64))
NEO4J_BATCH_SIZE    = int(os.getenv("NEO4J_BATCH_SIZE",    500))
TOP_K               = int(os.getenv("TOP_K",               10))
HOP_DEPTH           = int(os.getenv("HOP_DEPTH",           2))
CITES_THRESHOLD     = int(os.getenv("CITES_THRESHOLD",     2))
MAX_AUTHORS         = int(os.getenv("MAX_AUTHORS",         5))
MAX_CONCEPTS        = int(os.getenv("MAX_CONCEPTS",        10))
MIN_ABSTRACT_WORDS  = int(os.getenv("MIN_ABSTRACT_WORDS",  30))
