import os

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:4b"

GENERATION_BACKEND = "auto"
LEASE_COORDINATOR_URL = os.getenv(
    "LEASE_COORDINATOR_URL",
    "http://127.0.0.1:9000/chat/completions",
)
LEASE_MODEL = "gpt-oss-120b"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATABASE_DIR = os.path.abspath(
    os.getenv(
        "DATABASE_DIR",
        os.path.join(REPO_DIR, "..", "..", "DATABASE"),
    )
)
QDRANT_PATH = os.path.join(DATABASE_DIR, "qdrant_db")
INGEST_STATE_PATH = os.path.join(QDRANT_PATH, "ingest_state.json")
LEXICAL_INDEX_PATH = os.path.join(DATABASE_DIR, "lexical_index.pkl")
COLLECTION_NAME = "buku_islam"
VECTOR_DIM = 768

JSON_DIR = os.path.join(DATABASE_DIR, "json_output")
REFERENCE_DATA_DIR = os.getenv(
    "REFERENCE_DATA_DIR",
    os.path.join(DATABASE_DIR, "reference_data"),
)
QURAN_REFERENCE_PATH = os.getenv(
    "QURAN_REFERENCE_PATH",
    os.path.join(REFERENCE_DATA_DIR, "quran", "quran-uthmani.json"),
)
QURAN_TRANSLATION_PATH = os.getenv(
    "QURAN_TRANSLATION_PATH",
    os.path.join(REFERENCE_DATA_DIR, "quran", "translation-id.json"),
)
HADITH_REFERENCE_DIR = os.getenv(
    "HADITH_REFERENCE_DIR",
    os.path.join(REFERENCE_DATA_DIR, "hadith"),
)
DORAR_API_URL = os.getenv("DORAR_API_URL", "https://dorar.net/dorar_api.json")

DEFAULT_TOP_K = 5
RETRIEVAL_CANDIDATES = 20
RETRIEVAL_CANDIDATES_PER_QUERY = 12
QDRANT_SEARCH_HNSW_EF = int(os.getenv("QDRANT_SEARCH_HNSW_EF", "32"))
QDRANT_SEARCH_EXACT = os.getenv("QDRANT_SEARCH_EXACT", "0") == "1"
QDRANT_SEARCH_PAYLOAD_FIELDS = [
    "text",
    "title",
    "book_id",
    "filename",
    "page_start",
    "page_end",
    "chunk_idx",
    "language",
    "source_type",
    "document_type",
    "conversion_status",
]
CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP = 50
CHUNK_MIN_CHARS = 80

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "128"))
EMBED_RETRY_COUNT = int(os.getenv("EMBED_RETRY_COUNT", "3"))
EMBED_TIMEOUT = int(os.getenv("EMBED_TIMEOUT", "60"))
