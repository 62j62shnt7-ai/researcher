"""SQLite-backed document store: documents, chunks, FTS5 keyword index, vector index."""
import json
import re
import sqlite3
import threading
import time
from typing import Optional

import numpy as np

from .config import DB_PATH, DATA_DIR

_lock = threading.RLock()
_conn: Optional[sqlite3.Connection] = None

# In-memory vector cache: dict mapping dim -> (chunk_ids: list[int], matrix: np.ndarray normalized)
_vec_cache: dict[int, tuple[list[int], np.ndarray]] = {}


def get_conn() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _init_schema(_conn)
        return _conn


def _init_schema(c: sqlite3.Connection):
    c.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            format TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',  -- pending|parsing|embedding|ready|ready_keyword_only|error
            error TEXT,
            pages INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            added_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            idx INTEGER NOT NULL,
            text TEXT NOT NULL,
            location TEXT  -- e.g. "p. 12", "Sheet: Pipe Data", "Slide 4"
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text, content='chunks', content_rowid='id', tokenize='porter unicode61'
        );
        CREATE TABLE IF NOT EXISTS vectors (
            chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            dim INTEGER NOT NULL,
            embedding BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_msgs_chat ON chat_messages(chat_id);
        """
    )
    # migrations for databases created by older versions
    cols = [r[1] for r in c.execute("PRAGMA table_info(documents)").fetchall()]
    if "collection" not in cols:
        c.execute("ALTER TABLE documents ADD COLUMN collection TEXT")
    if "file_mtime" not in cols:
        c.execute("ALTER TABLE documents ADD COLUMN file_mtime REAL")
    c.commit()


def _invalidate_vec_cache():
    global _vec_cache
    _vec_cache.clear()


# ---------------- documents ----------------

def add_document(filename: str, stored_path: str, fmt: str) -> int:
    with _lock:
        c = get_conn()
        cur = c.execute(
            "INSERT INTO documents (filename, stored_path, format, status, added_at) VALUES (?,?,?,?,?)",
            (filename, stored_path, fmt, "pending", time.time()),
        )
        c.commit()
        return cur.lastrowid


def set_status(doc_id: int, status: str, error: str = None):
    with _lock:
        c = get_conn()
        c.execute("UPDATE documents SET status=?, error=? WHERE id=?", (status, error, doc_id))
        c.commit()


def set_doc_meta(doc_id: int, pages: int = None, chunk_count: int = None):
    with _lock:
        c = get_conn()
        if pages is not None:
            c.execute("UPDATE documents SET pages=? WHERE id=?", (pages, doc_id))
        if chunk_count is not None:
            c.execute("UPDATE documents SET chunk_count=? WHERE id=?", (chunk_count, doc_id))
        c.commit()


def list_documents() -> list[dict]:
    with _lock:
        c = get_conn()
        rows = c.execute("SELECT * FROM documents ORDER BY added_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_document(doc_id: int) -> Optional[dict]:
    with _lock:
        c = get_conn()
        r = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(r) if r else None


def delete_document(doc_id: int):
    with _lock:
        c = get_conn()
        ids = [r[0] for r in c.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()]
        if ids:
            qmarks = ",".join("?" * len(ids))
            c.execute(f"DELETE FROM vectors WHERE chunk_id IN ({qmarks})", ids)
            for cid in ids:
                c.execute("INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', ?, (SELECT text FROM chunks WHERE id=?))", (cid, cid))
            c.execute(f"DELETE FROM chunks WHERE id IN ({qmarks})", ids)
        c.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        c.commit()
        _invalidate_vec_cache()
def clear_all_documents():
    with _lock:
        c = get_conn()
        c.execute("DELETE FROM vectors")
        c.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        c.execute("DELETE FROM chunks")
        c.execute("DELETE FROM documents")
        c.commit()
        _invalidate_vec_cache()



def clear_chunks(doc_id: int):
    """Remove chunks/vectors for a doc (before re-indexing)."""
    with _lock:
        c = get_conn()
        ids = [r[0] for r in c.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()]
        if ids:
            qmarks = ",".join("?" * len(ids))
            c.execute(f"DELETE FROM vectors WHERE chunk_id IN ({qmarks})", ids)
            for cid in ids:
                c.execute("INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', ?, (SELECT text FROM chunks WHERE id=?))", (cid, cid))
            c.execute(f"DELETE FROM chunks WHERE id IN ({qmarks})", ids)
        c.commit()
        _invalidate_vec_cache()


# ---------------- chunks ----------------

def add_chunks(doc_id: int, chunks: list[dict]) -> list[int]:
    """chunks: [{text, location}] -> returns chunk ids"""
    out = []
    with _lock:
        c = get_conn()
        for i, ch in enumerate(chunks):
            cur = c.execute(
                "INSERT INTO chunks (doc_id, idx, text, location) VALUES (?,?,?,?)",
                (doc_id, i, ch["text"], ch.get("location")),
            )
            cid = cur.lastrowid
            c.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?,?)", (cid, ch["text"]))
            out.append(cid)
        c.commit()
    return out


def add_vectors(pairs: list[tuple[int, list[float]]]):
    with _lock:
        c = get_conn()
        for chunk_id, emb in pairs:
            arr = np.asarray(emb, dtype=np.float32)
            c.execute(
                "INSERT OR REPLACE INTO vectors (chunk_id, dim, embedding) VALUES (?,?,?)",
                (chunk_id, arr.shape[0], arr.tobytes()),
            )
        c.commit()
        _invalidate_vec_cache()


def get_chunk_by_idx(doc_id: int, idx: int) -> Optional[dict]:
    with _lock:
        c = get_conn()
        r = c.execute("SELECT * FROM chunks WHERE doc_id=? AND idx=?", (doc_id, idx)).fetchone()
        return dict(r) if r else None


def get_chunk(chunk_id: int) -> Optional[dict]:
    with _lock:
        c = get_conn()
        r = c.execute(
            """SELECT ch.*, d.filename FROM chunks ch JOIN documents d ON d.id = ch.doc_id WHERE ch.id=?""",
            (chunk_id,),
        ).fetchone()
        return dict(r) if r else None


# ---------------- search ----------------

def _fts_query(q: str) -> str:
    """Sanitize a free-text query into an FTS5 OR query of quoted tokens."""
    tokens = re.findall(r"[A-Za-z0-9_.\-]+", q)
    tokens = [t for t in tokens if t]
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens[:32])


def keyword_search(q: str, limit: int = 30, doc_ids: list[int] = None) -> list[dict]:
    with _lock:
        c = get_conn()
        sql = """
            SELECT ch.id, ch.doc_id, ch.idx, ch.text, ch.location, d.filename,
                   bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks ch ON ch.id = chunks_fts.rowid
            JOIN documents d ON d.id = ch.doc_id
            WHERE chunks_fts MATCH ?
        """
        params: list = [_fts_query(q)]
        if doc_ids:
            sql += f" AND ch.doc_id IN ({','.join('?'*len(doc_ids))})"
            params += doc_ids
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        try:
            rows = c.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        # bm25: lower is better -> convert to positive relevance
        out = []
        for r in rows:
            d = dict(r)
            d["score"] = -float(d["score"])
            out.append(d)
        return out


def _load_vec_cache(target_dim: Optional[int] = None):
    global _vec_cache
    if target_dim is not None and target_dim in _vec_cache:
        return _vec_cache[target_dim]

    with _lock:
        c = get_conn()
        if target_dim is not None:
            rows = c.execute("SELECT chunk_id, dim, embedding FROM vectors WHERE dim=?", (target_dim,)).fetchall()
        else:
            rows = c.execute("SELECT chunk_id, dim, embedding FROM vectors").fetchall()

        if not rows:
            return ([], None)

        dim = target_dim or rows[0]["dim"]
        ids, mats = [], []
        for r in rows:
            if r["dim"] == dim:
                ids.append(r["chunk_id"])
                mats.append(np.frombuffer(r["embedding"], dtype=np.float32))

        if not mats:
            return ([], None)

        mat = np.vstack(mats)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        cached = (ids, mat)
        _vec_cache[dim] = cached
        return cached


def vector_search(query_emb: list[float], limit: int = 30, doc_ids: list[int] = None) -> list[dict]:
    if not query_emb:
        return []
    target_dim = len(query_emb)
    ids, mat = _load_vec_cache(target_dim=target_dim)
    if mat is None or not ids:
        return []
    q = np.asarray(query_emb, dtype=np.float32)
    if q.shape[0] != mat.shape[1]:
        return []
    qn = np.linalg.norm(q)
    if qn == 0:
        return []
    sims = mat @ (q / qn)
    order = np.argsort(-sims)

    candidate_cids = []
    cid_to_score = {}
    for i in order:
        cid = ids[int(i)]
        candidate_cids.append(cid)
        cid_to_score[cid] = float(sims[int(i)])
        if len(candidate_cids) >= limit * 3:
            break

    if not candidate_cids:
        return []

    out = []
    allowed = set(doc_ids) if doc_ids else None
    with _lock:
        c = get_conn()
        qmarks = ",".join("?" * len(candidate_cids))
        sql = f"""
            SELECT ch.id, ch.doc_id, ch.idx, ch.text, ch.location, d.filename
            FROM chunks ch
            JOIN documents d ON d.id = ch.doc_id
            WHERE ch.id IN ({qmarks})
        """
        rows = c.execute(sql, candidate_cids).fetchall()
        rows_by_id = {r["id"]: dict(r) for r in rows}

        for cid in candidate_cids:
            if cid in rows_by_id:
                d = rows_by_id[cid]
                if allowed and d["doc_id"] not in allowed:
                    continue
                d["score"] = cid_to_score[cid]
                out.append(d)
                if len(out) >= limit:
                    break
    return out


def hybrid_search(q: str, query_emb: Optional[list[float]], top_k: int = 8,
                  vector_weight: float = 0.6, doc_ids: list[int] = None) -> list[dict]:
    """Reciprocal-rank-fusion of keyword and vector results."""
    kw = keyword_search(q, limit=top_k * 4, doc_ids=doc_ids)
    vec = vector_search(query_emb, limit=top_k * 4, doc_ids=doc_ids) if query_emb else []
    K = 60.0
    scores: dict[int, float] = {}
    items: dict[int, dict] = {}
    for rank, r in enumerate(kw):
        scores[r["id"]] = scores.get(r["id"], 0) + (1 - vector_weight) / (K + rank + 1)
        items[r["id"]] = r
    for rank, r in enumerate(vec):
        scores[r["id"]] = scores.get(r["id"], 0) + vector_weight / (K + rank + 1)
        items[r["id"]] = r
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    out = []
    for cid, s in ranked:
        d = dict(items[cid])
        d["score"] = s
        out.append(d)
    return out


def stats() -> dict:
    with _lock:
        c = get_conn()
        docs = c.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        vecs = c.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        return {"documents": docs, "chunks": chunks, "vectors": vecs}


# ---------------- chats (saved conversations) ----------------

def create_chat(title: str) -> int:
    with _lock:
        c = get_conn()
        now = time.time()
        cur = c.execute("INSERT INTO chats (title, created_at, updated_at) VALUES (?,?,?)",
                        (title[:80] or "New chat", now, now))
        c.commit()
        return cur.lastrowid


def list_chats() -> list[dict]:
    with _lock:
        c = get_conn()
        rows = c.execute(
            """SELECT ch.*, (SELECT COUNT(*) FROM chat_messages m WHERE m.chat_id = ch.id) AS messages
               FROM chats ch ORDER BY updated_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def rename_chat(chat_id: int, title: str):
    with _lock:
        c = get_conn()
        c.execute("UPDATE chats SET title=? WHERE id=?", (title[:80], chat_id))
        c.commit()


def delete_chat(chat_id: int):
    with _lock:
        c = get_conn()
        c.execute("DELETE FROM chat_messages WHERE chat_id=?", (chat_id,))
        c.execute("DELETE FROM chats WHERE id=?", (chat_id,))
        c.commit()


def get_chat_messages(chat_id: int) -> list[dict]:
    with _lock:
        c = get_conn()
        rows = c.execute(
            "SELECT role, content, sources FROM chat_messages WHERE chat_id=? ORDER BY id",
            (chat_id,),
        ).fetchall()
        out = []
        for r in rows:
            out.append({
                "role": r["role"],
                "content": r["content"],
                "sources": json.loads(r["sources"]) if r["sources"] else None,
            })
        return out


def add_chat_message(chat_id: int, role: str, content: str, sources=None):
    with _lock:
        c = get_conn()
        now = time.time()
        c.execute(
            "INSERT INTO chat_messages (chat_id, role, content, sources, created_at) VALUES (?,?,?,?,?)",
            (chat_id, role, content, json.dumps(sources, ensure_ascii=False) if sources else None, now),
        )
        c.execute("UPDATE chats SET updated_at=? WHERE id=?", (now, chat_id))
        c.commit()


# ---------------- collections & watch helpers ----------------

def set_collection(doc_id: int, name):
    with _lock:
        c = get_conn()
        c.execute("UPDATE documents SET collection=? WHERE id=?", (name or None, doc_id))
        c.commit()


def list_collections() -> list[str]:
    with _lock:
        c = get_conn()
        rows = c.execute(
            "SELECT DISTINCT collection FROM documents WHERE collection IS NOT NULL AND collection != '' ORDER BY collection"
        ).fetchall()
        return [r[0] for r in rows]


def get_doc_by_path(path: str):
    with _lock:
        c = get_conn()
        r = c.execute("SELECT * FROM documents WHERE stored_path=?", (path,)).fetchone()
        return dict(r) if r else None


def set_doc_mtime(doc_id: int, mtime: float):
    with _lock:
        c = get_conn()
        c.execute("UPDATE documents SET file_mtime=? WHERE id=?", (mtime, doc_id))
        c.commit()
