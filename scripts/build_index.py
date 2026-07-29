#!/usr/bin/env python3
"""
Build Index Script for Researcher AI
Parses engineering codes in codes/ directory, generates chunks, BM25 tokens,
and optional Gemini vector embeddings, saving the output to knowledge_base.json.
"""

import os
import sys
import json
import re
import math
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

# Helper for parsing PDF using pypdf or fallback
def extract_pdf_pages(file_path):
    pages = []
    try:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({"page": idx + 1, "text": text.strip()})
        if pages:
            return pages
    except Exception as e:
        print(f"pypdf extraction failed for {file_path}: {e}")

    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append({"page": idx + 1, "text": text.strip()})
        if pages:
            return pages
    except Exception as e:
        print(f"pdfplumber extraction failed for {file_path}: {e}")

    return pages

# Helper for parsing DOCX
def extract_docx_pages(file_path):
    try:
        import docx
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        text = "\n".join(full_text)
        chunks_text = [text[i:i+1000] for i in range(0, len(text), 1000)]
        return [{"page": idx + 1, "text": c} for idx, c in enumerate(chunks_text)]
    except Exception as e:
        print(f"DOCX extraction failed for {file_path}: {e}")
        return []

# Helper for parsing TXT / MD / CSV
def extract_text_pages(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        chunks_text = [text[i:i+1000] for i in range(0, len(text), 1000)]
        return [{"page": idx + 1, "text": c} for idx, c in enumerate(chunks_text)]
    except Exception as e:
        print(f"Text extraction failed for {file_path}: {e}")
        return []

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def chunk_text(pages, filename, max_chars=800, overlap=100):
    chunks = []
    chunk_id = 0
    
    for page_info in pages:
        page_num = page_info["page"]
        text = page_info["text"]
        if not text:
            continue
            
        start = 0
        while start < len(text):
            end = start + max_chars
            chunk_str = text[start:end].strip()
            if chunk_str:
                clause_match = re.search(r'(?:(?:para|section|clause|article|part)\s+[\d\.]+|[\d]+\.[\d]+(?:\.[\d]+)?)', chunk_str, re.IGNORECASE)
                clause = clause_match.group(0) if clause_match else f"Page {page_num}"
                
                chunks.append({
                    "id": f"{filename}_{chunk_id}",
                    "file": filename,
                    "page": page_num,
                    "clause": clause,
                    "text": chunk_str,
                    "tokens": tokenize(chunk_str)
                })
                chunk_id += 1
            start += (max_chars - overlap)
            
    return chunks

def discover_embedding_model(api_key):
    if not api_key:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        req = urllib.request.Request(url, headers={"x-goog-api-key": api_key})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            models = data.get("models", [])
            for m in models:
                m_name = m.get("name", "")
                methods = m.get("supportedGenerationMethods", [])
                if "embedContent" in methods and "embedding-001" not in m_name:
                    clean_name = m_name.replace("models/", "")
                    print(f"Discovered working embedding model for your key: {clean_name}")
                    return clean_name
    except Exception as e:
        print(f"Could not query models list: {e}")

    return "gemini-embedding-2"

def fetch_gemini_embedding(text, api_key, model_name):
    if not api_key or not model_name:
        return []
    if "embedding-001" in model_name:
        model_name = "gemini-embedding-2"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:embedContent"
    payload = {
        "model": f"models/{model_name}",
        "content": {
            "parts": [{"text": text[:8000]}]
        }
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
    try:
        with urllib.request.urlopen(req) as resp:
            res_json = json.loads(resp.read().decode('utf-8'))
            return res_json.get("embedding", {}).get("values", [])
    except Exception as e:
        return []


def build_knowledge_base():
    codes_dir = Path("codes")
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    working_embed_model = None
    if api_key:
        working_embed_model = discover_embedding_model(api_key)
        
    if not codes_dir.exists():
        print("No codes/ directory found. Creating empty codes/ directory...")
        codes_dir.mkdir(exist_ok=True)
        
    supported_exts = {".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md"}
    all_files = [f for f in codes_dir.glob("*") if f.suffix.lower() in supported_exts]
    
    print(f"Found {len(all_files)} document(s) in codes/")
    
    documents_meta = []
    all_chunks = []
    embedding_failed_count = 0
    
    local_embedder = None
    try:
        from fastembed import TextEmbedding
        print("Using free local FastEmbed engine (BAAI/bge-small-en-v1.5) for GitHub indexing...")
        local_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    except Exception as e:
        print(f"FastEmbed local engine not initialized: {e}")

    for file_path in all_files:
        filename = file_path.name
        ext = file_path.suffix.lower()
        print(f"Processing: {filename}...")

        if ext == ".pdf":
            pages = extract_pdf_pages(file_path)
        elif ext == ".docx":
            pages = extract_docx_pages(file_path)
        else:
            pages = extract_text_pages(file_path)

        doc_chunks = chunk_text(pages, filename)

        documents_meta.append({
            "filename": filename,
            "page_count": len(pages),
            "chunk_count": len(doc_chunks)
        })

        for c in doc_chunks:
            all_chunks.append(c)

    if local_embedder and all_chunks:
        print(f"Computing free local CPU embeddings for {len(all_chunks)} chunk(s)...")
        try:
            chunk_texts = [c["text"] for c in all_chunks]
            embs = list(local_embedder.embed(chunk_texts))
            for idx, vec in enumerate(embs):
                all_chunks[idx]["embedding"] = vec.tolist()
            print("Successfully embedded all document chunks using free local FastEmbed!")
        except Exception as e:
            print(f"Local embedding computation failed: {e}")

    if all_chunks and not all_chunks[0].get("embedding") and api_key and working_embed_model:
        print("Falling back to Gemini API for building vector embeddings...")
        for c in all_chunks:
            vec = fetch_gemini_embedding(c["text"], api_key, working_embed_model)
            if vec:
                c["embedding"] = vec
            else:
                embedding_failed_count += 1
                if embedding_failed_count > 3:
                    working_embed_model = None
                    print("Embedding calls failing on key. Falling back to BM25 for remaining chunks.")

    kb_data = {
        "documents": documents_meta,
        "chunks": all_chunks,
        "generated_at": str(Path("codes").stat().st_mtime) if codes_dir.exists() else ""
    }

    out_file = out_dir / "knowledge_base.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, indent=2)

    root_out_file = Path("knowledge_base.json")
    with open(root_out_file, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, indent=2)

    print(f"Successfully generated knowledge_base.json with {len(all_chunks)} chunks from {len(documents_meta)} files.")


if __name__ == "__main__":
    build_knowledge_base()
