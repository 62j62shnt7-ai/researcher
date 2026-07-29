"""Embeddings via any OpenAI-compatible /v1/embeddings endpoint (LM Studio, Ollama, OpenAI...)."""
import httpx


class EmbeddingError(Exception):
    pass


def embed_texts(texts: list[str], cfg: dict) -> list[list[float]]:
    """cfg = config['embeddings']. Raises EmbeddingError on failure."""
    base = cfg.get("base_url", "").rstrip("/")
    if not base:
        raise EmbeddingError("No embeddings base URL configured")
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    payload = {"input": texts}
    if cfg.get("model"):
        payload["model"] = cfg["model"]
    try:
        with httpx.Client(timeout=120) as client:
            r = client.post(f"{base}/embeddings", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        raise EmbeddingError(f"Embeddings request failed: {e}") from e
    try:
        items = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [it["embedding"] for it in items]
    except (KeyError, TypeError) as e:
        raise EmbeddingError(f"Unexpected embeddings response: {e}") from e


def embed_in_batches(texts: list[str], cfg: dict, batch_size: int = 32, progress=None) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        out.extend(embed_texts(batch, cfg))
        if progress:
            progress(min(i + batch_size, len(texts)), len(texts))
    return out


def list_models(base_url: str, api_key: str = "") -> list[str]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    with httpx.Client(timeout=15) as client:
        r = client.get(f"{base_url.rstrip('/')}/models", headers=headers)
        r.raise_for_status()
        data = r.json()
    return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
