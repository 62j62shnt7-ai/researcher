"""Embeddings via Local CPU (FastEmbed / sentence-transformers), Google Gemini, or OpenAI-compatible endpoints."""
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


def _embed_gemini(texts: list[str], cfg: dict) -> list[list[float]]:
    api_key = cfg.get("api_key", "")
    if not api_key:
        raise EmbeddingError("Gemini embedding requires an API key. Please set your API key in Settings.")

    base = (cfg.get("base_url") or "https://generativelanguage.googleapis.com").rstrip("/")
    if base.endswith("/v1beta") or base.endswith("/v1") or base.endswith("/openai"):
        base = "https://generativelanguage.googleapis.com"

    headers_native = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }

    # Model resolution: try preferred model first, then known fallbacks
    user_model = (cfg.get("model") or "gemini-embedding-2").replace("models/", "")
    if "embedding-001" in user_model:
        user_model = "gemini-embedding-2"
    candidate_models = [user_model]
    for fb in ("gemini-embedding-2", "text-embedding-004"):
        if fb not in candidate_models:
            candidate_models.append(fb)

    last_err = None
    sub_batch_size = 10

    with httpx.Client(timeout=60) as client:
        for m_name in candidate_models:
            model_path = f"models/{m_name}"
            batch_url = f"{base}/v1beta/{model_path}:batchEmbedContents"
            all_embeddings = []
            model_failed = False

            # Try Method 1: batchEmbedContents
            try:
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
                            time.sleep(2.5 * (attempt + 1))
                            continue
                        elif r.status_code == 404:
                            # Model not found on this endpoint/tier, break to next candidate model
                            model_failed = True
                            last_err = f"Model {m_name} returned 404"
                            break
                        else:
                            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                            break

                    if model_failed or not sub_success:
                        model_failed = True
                        break

                    time.sleep(0.3)

                if not model_failed and len(all_embeddings) == len(texts):
                    return all_embeddings
            except Exception as e:
                last_err = e
                model_failed = True

            # If batch endpoint had a 404, try next candidate model
            if model_failed and "404" in str(last_err):
                continue

            # Method 2: Single embedContent per text snippet with backoff retry
            embed_url = f"{base}/v1beta/{model_path}:embedContent"
            results = []
            single_failed = False
            try:
                for t in texts:
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
                            time.sleep(3.0 * (attempt + 1))
                            continue
                        elif r.status_code == 404:
                            single_failed = True
                            break
                        else:
                            break

                    if not sub_ok:
                        single_failed = True
                        break

                    time.sleep(0.3)

                if not single_failed and len(results) == len(texts):
                    return results
            except Exception as e:
                last_err = e

    raise EmbeddingError(f"Gemini embedding failed: {last_err or 'unknown error'}")


def embed_texts(texts: list[str], cfg: dict) -> list[list[float]]:
    """cfg = config['embeddings']. Raises EmbeddingError on failure."""
    if not texts:
        return []

    provider = cfg.get("provider")
    base = cfg.get("base_url", "").rstrip("/")
    api_key = cfg.get("api_key", "")

    if provider == "local":
        try:
            return _embed_local(texts, cfg)
        except EmbeddingError as local_err:
            # If local CPU engine is unavailable but an API key is available, fall back to Gemini
            if api_key or "generativelanguage.googleapis.com" in base:
                return _embed_gemini(texts, cfg)
            raise local_err

    if provider == "gemini" or "generativelanguage.googleapis.com" in base or (api_key and "localhost" not in base and "127.0.0.1" not in base):
        try:
            return _embed_gemini(texts, cfg)
        except EmbeddingError as gemini_err:
            # If Gemini fails (e.g. offline or quota), attempt local CPU fallback
            try:
                return _embed_local(texts, cfg)
            except Exception:
                raise gemini_err

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


def embed_in_batches(texts: list[str], cfg: dict, batch_size: int = 32, progress=None) -> list[list[float]]:
    out = []
    effective_batch = max(1, min(batch_size, 32))
    for i in range(0, len(texts), effective_batch):
        batch = texts[i:i + effective_batch]
        out.extend(embed_texts(batch, cfg))
        if progress:
            progress(min(i + effective_batch, len(texts)), len(texts))
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
