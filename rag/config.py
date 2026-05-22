import os

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:4b"

GENERATION_BACKEND = "auto"
LEASE_COORDINATOR_URL = "http://127.0.0.1:9000/chat/completions"
LEASE_MODEL = "gpt-oss-120b"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
QDRANT_PATH = os.path.join(REPO_DIR, "qdrant_db")
INGEST_STATE_PATH = os.path.join(QDRANT_PATH, "ingest_state.json")
COLLECTION_NAME = "buku_islam"
VECTOR_DIM = 768

JSON_DIR = os.path.join(REPO_DIR, "json_output")

DEFAULT_TOP_K = 5
RETRIEVAL_CANDIDATES = 20
RETRIEVAL_CANDIDATES_PER_QUERY = 12
CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP = 50
CHUNK_MIN_CHARS = 80

EMBED_BATCH_SIZE = 64
INGEST_BATCH_SIZE = 128
EMBED_RETRY_COUNT = 3
EMBED_TIMEOUT = 60
