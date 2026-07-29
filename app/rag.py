"""Retrieval-augmented chat: query expansion, hybrid retrieval, neighbor context, prompt building."""
import re

from . import store
from .embeddings import embed_texts, EmbeddingError
from .providers import complete, ProviderError

SYSTEM_PROMPT = """You are an expert engineering assistant with deep knowledge of codes, standards, and engineering practice. You answer questions using excerpts retrieved from the user's document library, shown in SOURCES below.

How to answer:
1. Start with a direct answer to the question.
2. Then EXPLAIN it properly: what it means in practice, why the requirement exists where evident, what conditions/exceptions apply, and how the pieces relate. Do not just quote fragments back — interpret and synthesize them like a senior engineer explaining to a colleague.
3. Combine information across multiple sources when they cover the same topic; point out when sources differ or when a requirement in one place is modified by another.
4. Quote exact clause numbers, values, formulas, tolerances, and table data when present. Never invent clause numbers or values.
5. Cite sources inline with bracketed numbers, e.g. [1] or [2][3], so the user can verify.
6. If the sources only partially answer the question, answer what you can from them, then clearly separate any additional general engineering knowledge with "(general knowledge, not from your documents)". If the sources contain nothing relevant, say so plainly.
7. Use markdown (headings, tables, lists) when it makes the answer clearer.

SOURCES:
{sources}"""

REWRITE_PROMPT = """You generate search queries for a technical document library (engineering codes, standards, specifications).

Given the conversation and the user's latest question, write 3 different search queries that would find the relevant passages. Make them complementary:
- one with the key technical terms of the question (expand abbreviations)
- one rephrased with synonyms or related standard terminology
- one targeting any specific clause/section/table numbers or standard designations mentioned

Resolve pronouns and references using the conversation (e.g. "what about its tolerance?" -> name the actual subject).
Output ONLY the 3 queries, one per line, no numbering, no explanation."""


def _make_queries(history: list[dict], question: str, chat_cfg: dict) -> list[str]:
    """Use the LLM to expand the question into better search queries. Falls back to the raw question."""
    convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])
    try:
        text = complete(
            [{"role": "system", "content": REWRITE_PROMPT},
             {"role": "user", "content": f"Conversation:\n{convo}\n\nLatest question: {question}"}],
            chat_cfg, max_tokens=200,
        )
        queries = [q.strip(" -•\t") for q in text.splitlines() if len(q.strip()) > 3]
        queries = [re.sub(r'^\d+[.)]\s*', '', q) for q in queries][:3]
    except Exception:
        queries = []
    if question not in queries:
        queries.insert(0, question)
    return queries[:4]


def _expand_with_neighbors(hit: dict, max_chars: int) -> str:
    """Prepend/append adjacent chunks from the same document so the model sees continuous text."""
    text = hit["text"]
    idx = hit.get("idx")
    if idx is None:
        return text[:max_chars]
    budget = max_chars - len(text)
    if budget > 300:
        nxt = store.get_chunk_by_idx(hit["doc_id"], idx + 1)
        if nxt:
            text = text + "\n" + nxt["text"][: budget // 2]
        prev = store.get_chunk_by_idx(hit["doc_id"], idx - 1)
        if prev:
            text = prev["text"][-(budget // 2):] + "\n" + text
    return text[:max_chars]


def retrieve(question: str, config: dict, doc_ids: list[int] = None,
             history: list[dict] = None) -> list[dict]:
    rcfg = config["retrieval"]
    ecfg = config["embeddings"]
    top_k = rcfg.get("top_k", 10)

    # 1. build search queries (LLM expansion + raw question)
    if rcfg.get("query_expansion", True) and history:
        queries = _make_queries(history, question, config["chat"])
    else:
        queries = [question]

    # 2. run hybrid search per query, fuse with reciprocal-rank fusion
    fused: dict[int, float] = {}
    items: dict[int, dict] = {}
    K = 60.0
    for qi, q in enumerate(queries):
        q_emb = None
        if ecfg.get("enabled", True):
            try:
                q_emb = embed_texts([q], ecfg)[0]
            except EmbeddingError:
                q_emb = None
        results = store.hybrid_search(
            q, q_emb, top_k=top_k * 2,
            vector_weight=rcfg.get("vector_weight", 0.6),
            doc_ids=doc_ids,
        )
        weight = 1.0 if qi == 0 else 0.8  # raw question counts slightly more
        for rank, r in enumerate(results):
            fused[r["id"]] = fused.get(r["id"], 0) + weight / (K + rank + 1)
            items[r["id"]] = r

    ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]

    # 3. expand each hit with neighboring chunks for continuous context
    max_chars = rcfg.get("max_source_chars", 3500)
    out = []
    for cid, score in ranked:
        hit = dict(items[cid])
        hit["text"] = _expand_with_neighbors(hit, max_chars)
        hit["score"] = score
        out.append(hit)
    return out


def fit_sources(sources: list[dict], config: dict, history: list[dict] = None) -> list[dict]:
    """Trim the source list so the full prompt fits the model's context window.
    Prevents silent truncation by the server, which often yields empty answers.
    Rough budget: (context_tokens - output reserve - instructions - history) * 4 chars/token."""
    ccfg = config.get("chat", {})
    ctx = int(ccfg.get("context_tokens", 8192) or 8192)
    out_reserve = int(ccfg.get("max_tokens", 3000) or 3000)
    hist_chars = sum(len(m.get("content", "")) for m in (history or [])[-12:])
    budget = max(3000, (ctx - out_reserve - 700) * 4 - hist_chars)

    kept, used = [], 0
    for s in sources:
        remaining = budget - used
        if remaining <= 0:
            break
        s = dict(s)
        if len(s["text"]) > remaining:
            if remaining < 500 and kept:
                break  # don't add a uselessly tiny fragment
            s["text"] = s["text"][:remaining]
        used += len(s["text"])
        kept.append(s)
    return kept


COMPARE_NOTE = """

COMPARISON MODE: the user is comparing the requirements of several documents. Structure your answer as:
1. One-paragraph summary of the key difference(s).
2. A markdown comparison table - rows are the technical aspects, one column per document - quoting exact values/clauses with citations.
3. Bullet notes on conflicts, gaps (topics one document covers that another does not), and which requirement is more stringent."""


def build_messages(history: list[dict], sources: list[dict], compare: bool = False) -> list[dict]:
    """history: full chat history [{role, content}]. Last user message is the question."""
    lines = []
    for i, s in enumerate(sources, 1):
        loc = f", {s['location']}" if s.get("location") else ""
        lines.append(f"[{i}] ({s['filename']}{loc})\n{s['text']}")
    src_text = "\n\n---\n\n".join(lines) if lines else "(no matching excerpts found in the library)"
    system = SYSTEM_PROMPT.format(sources=src_text)
    if compare:
        system += COMPARE_NOTE
    return [{"role": "system", "content": system}] + history[-12:]
