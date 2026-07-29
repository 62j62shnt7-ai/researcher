"""Embeddings via any OpenAI-compatible /v1/embeddings endpoint (LM Studio, Ollama, OpenAI...)."""
import time
import httpx



class EmbeddingError(Exception):
    pass


_local_embedding_model = None

def _get_local_model():
    global _local_embedding_model
    if _local_embedding_model is not None:
        return _local_embedding_model
    try:
        from fastembed import TextEmbedding
        _local_embedding_model = ("fastembed", TextEmbedding(model_name="BAAI/bge-small-en-v1.5"))
        return _local_embedding_model
    except ImportError:
        pass

    try:
        from sentence_transformers import SentenceTransformer
        _local_embedding_model = ("sentence-transformers", SentenceTransformer("all-MiniLM-L6-v2"))
        return _local_embedding_model
    except ImportError:
        pass

    raise EmbeddingError(
        "Local CPU embeddings engine not installed. Click 'Check & update dependencies' in Settings "
        "or run 'pip install fastembed' in your terminal."
    )


def _embed_local(texts: list[str], cfg: dict) -> list[list[float]]:
    backend, model = _get_local_model()
    try:
        if backend == "fastembed":
            return [e.tolist() for e in model.embed(texts)]
        elif backend == "sentence-transformers":
            embs = model.encode(texts, convert_to_numpy=True)
            return [e.tolist() for e in embs]
    except Exception as e:
        raise EmbeddingError(f"Local CPU embedding failed: {e}") from e


def embed_texts(texts: list[str], cfg: dict) -> list[list[float]]:
    """cfg = config['embeddings']. Raises EmbeddingError on failure."""
    provider = cfg.get("provider")
    base = cfg.get("base_url", "").rstrip("/")
    api_key = cfg.get("api_key", "")

    if provider == "local":
        return _embed_local(texts, cfg)

    if provider == "gemini" or "generativelanguage.googleapis.com" in base or (api_key and "localhost" not in base and "127.0.0.1" not in base):
        return _embed_gemini(texts, cfg)

    if not base:
        raise EmbeddingError("No embeddings base URL configured")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
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



def _embed_gemini(texts: list[str], cfg: dict) -> list[list[float]]:
    api_key = cfg.get("api_key", "")
    if not api_key:
        raise EmbeddingError("Gemini embedding requires an API key. Please set your API key in Settings.")

    raw_model = cfg.get("model") or "gemini-embedding-2"
    if "embedding-001" in raw_model:
        raw_model = "gemini-embedding-2"
    model_name = raw_model.replace("models/", "")
    model_path = f"models/{model_name}"

    base = (cfg.get("base_url") or "https://generativelanguage.googleapis.com").rstrip("/")
    if base.endswith("/v1beta") or base.endswith("/v1") or base.endswith("/openai"):
        base = "https://generativelanguage.googleapis.com"

    headers_native = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }

    # Method 1: Sub-batched batchEmbedContents endpoint (batches of 10 with rate-limit pacing)
    batch_url = f"{base}/v1beta/{model_path}:batchEmbedContents"
    sub_batch_size = 10
    all_embeddings = []
    failed_subbatch = False

    try:
        with httpx.Client(timeout=60) as client:
            for i in range(0, len(texts), sub_batch_size):
                sub_texts = texts[i:i + sub_batch_size]
                requests = [{"model": model_path, "content": {"parts": [{"text": t[:8000]}]}} for t in sub_texts]

                sub_success = False
                for attempt in range(4):
                    r = client.post(batch_url, headers=headers_native, json={"requests": requests})
                    if r.status_code == 200:
                        data = r.json()
                        embs = data.get("embeddings", [])
                        if embs and len(embs) == len(sub_texts):
                            all_embeddings.extend([e.get("values", []) for e in embs])
                            sub_success = True
                            break
                    elif r.status_code == 429:
                        time.sleep(3.0 * (attempt + 1))
                        continue
                    else:
                        break

                if not sub_success:
                    failed_subbatch = True
                    break

                time.sleep(0.4)  # pacing delay between sub-batches

        if not failed_subbatch and len(all_embeddings) == len(texts):
            return all_embeddings
    except Exception:
        pass

    # Method 2: Single embedContent endpoint per text snippet with backoff retry
    embed_url = f"{base}/v1beta/{model_path}:embedContent"
    results = []
    last_err = None
    try:
        with httpx.Client(timeout=60) as client:
            for idx, t in enumerate(texts):
                payload = {
                    "model": model_path,
                    "content": {"parts": [{"text": t[:8000]}]}
                }
                sub_ok = False
                for attempt in range(4):
                    r = client.post(embed_url, headers=headers_native, json=payload)
                    if r.status_code == 200:
                        data = r.json()
                        vals = data.get("embedding", {}).get("values", [])
                        if not vals:
                            candidates = data.get("embeddings", [])
                            if candidates:
                                vals = candidates[0].get("values", [])
                        if vals:
                            results.append(vals)
                            sub_ok = True
                            break
                    elif r.status_code == 429:
                        time.sleep(4.0 * (attempt + 1))
                        continue
                    else:
                        raise EmbeddingError(f"HTTP {r.status_code}: {r.text[:200]}")

                if not sub_ok:
                    raise EmbeddingError("Gemini API rate limit (HTTP 429) exceeded. Free API keys have a 15 RPM limit. Keyword search is active.")

                time.sleep(0.5)  # 500ms pacing delay between requests

        if len(results) == len(texts):
            return results
    except Exception as e:
        last_err = e

    raise EmbeddingError(f"Gemini embedding failed: {last_err}")










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
