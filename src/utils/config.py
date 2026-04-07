import os
from dotenv import load_dotenv

load_dotenv()

# ── Neo4j ─────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ── Groq ──────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

# ── Data paths ────────────────────────────────────────
CHROMA_DIR  = os.getenv("CHROMA_DIR",  "./data/chromadb")
RAW_FILE    = os.getenv("RAW_FILE",    "./data/arxiv_raw.json")
CLEAN_FILE  = os.getenv("CLEAN_FILE",  "./data/arxiv_clean.json")
NER_FILE    = os.getenv("NER_FILE",    "./data/arxiv_ner.json")

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
