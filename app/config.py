"""Configuration management. Stored in data/config.json next to the app."""
import json
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FILES_DIR = DATA_DIR / "files"
DB_PATH = DATA_DIR / "library.db"
CONFIG_PATH = DATA_DIR / "config.json"

DEFAULTS = {
    "chat": {
        "provider": "openai-compatible",   # "openai-compatible" | "anthropic"
        "base_url": "http://localhost:1234/v1",
        "api_key": "",
        "model": "",
        "temperature": 0.2,
        "max_tokens": 8192,
        "context_tokens": 32768,   # ample room for modern large-context LLMs
    },
    "embeddings": {
        "enabled": True,
        "base_url": "http://localhost:1234/v1",
        "api_key": "",
        "model": "",
        "batch_size": 32,
    },
    "watch": {
        "folders": [],        # folders scanned for new/changed documents
        "interval": 60,       # seconds between scans
    },
    "retrieval": {
        "top_k": 10,
        "chunk_size": 1500,
        "chunk_overlap": 200,
        "vector_weight": 0.6,        # hybrid blend: 0 = keyword only, 1 = vector only
        "query_expansion": True,     # use the LLM to generate better search queries
        "max_source_chars": 3500,    # context given to the model per source (with neighbors)
    },
}

_lock = threading.Lock()


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict:
    with _lock:
        if CONFIG_PATH.exists():
            try:
                user = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception:
                user = {}
        else:
            user = {}
        return _merge(DEFAULTS, user)


def save_config(cfg: dict) -> dict:
    current = load_config()          # preserve sections not included in this update
    merged = _merge(current, cfg)
    with _lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)
