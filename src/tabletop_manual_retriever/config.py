import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

UPLOADS_DIR = Path(
    os.environ.get("UPLOADS_DIR", PROJECT_ROOT / "data" / "uploads")
).resolve()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "manual_chunks")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


def _int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


CHUNK_MAX_CHARS = _int_env("CHUNK_MAX_CHARS", 1200)
CHUNK_OVERLAP_CHARS = _int_env("CHUNK_OVERLAP_CHARS", 150)
RAG_TOP_K = _int_env("RAG_TOP_K", 5)

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").strip().rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "").strip()
LLM_TIMEOUT_SECONDS = _int_env("LLM_TIMEOUT_SECONDS", 60)
