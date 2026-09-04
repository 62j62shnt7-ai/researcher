"""Chat LLM providers: OpenAI-compatible (LM Studio, Ollama, OpenAI, Groq, ...) and Anthropic.

All providers expose: stream_chat(messages, cfg) -> generator of text deltas.
messages: [{"role": "system"|"user"|"assistant", "content": str}]
"""
import json

import httpx


class ProviderError(Exception):
    pass


def stream_chat(messages: list[dict], cfg: dict):
    provider = cfg.get("provider", "openai-compatible")
    if provider == "anthropic":
        yield from _stream_anthropic(messages, cfg)
    elif provider == "gemini":
        yield from _stream_gemini(messages, cfg)
    else:
        yield from _stream_openai(messages, cfg)


def _stream_openai(messages: list[dict], cfg: dict):
    base = cfg.get("base_url", "").rstrip("/")
    if not base:
        raise ProviderError("No chat base URL configured. Open Settings.")
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    payload = {
        "messages": messages,
        "stream": True,
        "temperature": cfg.get("temperature", 0.2),
        "max_tokens": cfg.get("max_tokens", 3000),
    }
    if cfg.get("model"):
        payload["model"] = cfg["model"]
    try:
        with httpx.Client(timeout=httpx.Timeout(300, connect=15)) as client:
            with client.stream("POST", f"{base}/chat/completions", json=payload, headers=headers) as r:
                if r.status_code >= 400:
                    body = r.read().decode("utf-8", errors="replace")[:500]
                    raise ProviderError(f"Chat provider returned {r.status_code}: {body}")
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0].get("delta", {}).get("content")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
    except httpx.ConnectError as e:
        raise ProviderError(
            f"Cannot reach chat provider at {base}. Is LM Studio (or your API server) running?"
        ) from e
    except httpx.HTTPError as e:
        raise ProviderError(f"Chat request failed: {e}") from e


def _stream_anthropic(messages: list[dict], cfg: dict):
    api_key = cfg.get("api_key", "")
    if not api_key:
        raise ProviderError("Anthropic provider requires an API key. Open Settings.")
    base = (cfg.get("base_url") or "https://api.anthropic.com").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = [m for m in messages if m["role"] != "system"]
    payload = {
        "model": cfg.get("model") or "claude-haiku-4-5-20251001",
        "max_tokens": cfg.get("max_tokens", 3000),
        "temperature": cfg.get("temperature", 0.2),
        "stream": True,
        "messages": convo,
    }
    if system:
        payload["system"] = system
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=httpx.Timeout(300, connect=15)) as client:
            with client.stream("POST", f"{base}/v1/messages", json=payload, headers=headers) as r:
                if r.status_code >= 400:
                    body = r.read().decode("utf-8", errors="replace")[:500]
                    raise ProviderError(f"Anthropic returned {r.status_code}: {body}")
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    try:
                        obj = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") == "content_block_delta":
                        delta = obj.get("delta", {}).get("text")
                        if delta:
                            yield delta
    except httpx.HTTPError as e:
        raise ProviderError(f"Anthropic request failed: {e}") from e


def _stream_gemini(messages: list[dict], cfg: dict):
    api_key = cfg.get("api_key", "")
    if not api_key:
        raise ProviderError("Google Gemini provider requires an API key. Open Settings.")
    requested_model = cfg.get("model") or "gemini-3-flash-preview"
    base = (cfg.get("base_url") or "https://generativelanguage.googleapis.com").rstrip("/")
    if base.endswith("/v1beta") or base.endswith("/v1"):
        base = base.rsplit("/", 1)[0]

    system_text = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    models_to_try = [requested_model]
    for fb in ("gemini-3-flash-preview", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-flash-latest", "gemma-4-31b-it"):
        if fb not in models_to_try:
            models_to_try.append(fb)

    last_error = None
    headers = {"Content-Type": "application/json"}

    with httpx.Client(timeout=httpx.Timeout(120, connect=10)) as client:
        for model in models_to_try:
            gen_cfg = {
                "temperature": cfg.get("temperature", 0.2),
                "maxOutputTokens": cfg.get("max_tokens", 8192),
            }
            if cfg.get("deep_reasoning", True) and ("thinking" in model or "flash" in model):
                gen_cfg["thinkingConfig"] = {"thinkingBudget": 2048}

            payload = {
                "contents": contents,
                "generationConfig": gen_cfg
            }
            if system_text:
                payload["systemInstruction"] = {"parts": [{"text": system_text}]}

            url = f"{base}/v1beta/models/{model}:streamGenerateContent?alt=sse&key={api_key}"

            try:
                r = client.post(url, json=payload, headers=headers)
                if r.status_code == 400 and "thinkingConfig" in gen_cfg:
                    del gen_cfg["thinkingConfig"]
                    payload["generationConfig"] = gen_cfg
                    r = client.post(url, json=payload, headers=headers)

                if r.status_code in (404, 429, 503):
                    body = r.read().decode("utf-8", errors="replace")[:300]
                    last_error = ProviderError(f"Model {model} returned {r.status_code}: {body}")
                    continue

                if r.status_code >= 400:
                    body = r.read().decode("utf-8", errors="replace")[:500]
                    raise ProviderError(f"Gemini API returned {r.status_code}: {body}")

                streamed_any = False
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    try:
                        obj = json.loads(data_str)
                        candidates = obj.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for p in parts:
                                if "text" in p:
                                    streamed_any = True
                                    yield p["text"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

                if streamed_any:
                    return
            except httpx.HTTPError as e:
                last_error = ProviderError(f"Gemini request failed for {model}: {e}")
                continue

    if last_error:
        raise last_error
    raise ProviderError("All Gemini models were unavailable or returned errors.")


def complete(messages: list[dict], cfg: dict, max_tokens: int = 300) -> str:
    """Non-streaming convenience call (used for query rewriting)."""
    c = dict(cfg)
    c["max_tokens"] = max_tokens
    c["temperature"] = 0.0
    return "".join(stream_chat(messages, c))


def test_connection(cfg: dict) -> dict:
    """Returns {ok, models?, error?} for the configured chat provider."""
    provider = cfg.get("provider", "openai-compatible")
    try:
        if provider == "anthropic":
            if not cfg.get("api_key"):
                return {"ok": False, "error": "API key required"}
            return {"ok": True, "models": [
                "claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"
            ]}
        if provider == "gemini":
            api_key = cfg.get("api_key", "")
            if not api_key:
                return {"ok": False, "error": "API key required for Gemini"}
            base = (cfg.get("base_url") or "https://generativelanguage.googleapis.com").rstrip("/")
            if base.endswith("/v1beta") or base.endswith("/v1"):
                base = base.rsplit("/", 1)[0]
            url = f"{base}/v1beta/models?key={api_key}"
            with httpx.Client(timeout=15) as client:
                r = client.get(url)
                if r.status_code >= 400:
                    return {"ok": False, "error": f"Gemini API error: {r.status_code}"}
                data = r.json()
                models = [m.get("name", "").replace("models/", "") for m in data.get("models", [])
                          if "generateContent" in m.get("supportedGenerationMethods", [])]
                return {"ok": True, "models": models or ["gemini-3-flash-preview", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-flash-latest"]}

        from .embeddings import list_models
        models = list_models(cfg.get("base_url", ""), cfg.get("api_key", ""))
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e)}

