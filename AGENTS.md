# Researcher AI — Agent Guidelines & Architecture

This document outlines the architecture, data flows, and development standards for AI agents working in this repository to prevent errors, avoid quota exhaustion, and maintain system stability.

---

## 1. Project Overview & Tech Stack
* **Architecture:** Mobile-first, serverless Retrieval-Augmented Generation (RAG) web application.
* **Frontend Web App:** Vanilla HTML5, CSS3 (`styles.css`), and ES6+ JavaScript (`app.js`). Hosted on **GitHub Pages**.
* **AI & LLM Integration:** Google Gemini API (`gemini-1.5-flash` / `gemini-2.0-flash` for answers, `text-embedding-004` for semantic search).
* **Storage & Indexing:**
  * Global Library: `knowledge_base.json` (built from `codes/` PDFs by GitHub Actions / Python scripts).
  * Client-Side Library: Browser **IndexedDB** for local device-uploaded documents.
* **Search Engine:** Hybrid Sub-Millisecond RAG (BM25 keyword search for standard/clause numbers + Vector Cosine Similarity).

---

## 2. Directory Structure Map
```
researcher/
├── index.html                   # Main application UI layout
├── app.js                       # Client-side RAG engine, Gemini API client, IndexedDB store
├── styles.css                   # Custom responsive CSS with mobile-first design tokens
├── knowledge_base.json          # Pre-computed document chunks and vector embeddings
├── scripts/
│   ├── build_index.py           # Document parsing, chunking, and Gemini embedding generator
│   └── extract_text.py          # PDF/DOCX text extraction pipeline
├── codes/                       # Directory where engineering PDFs/standards are placed
├── .github/
│   └── workflows/
│       └── deploy.yml           # GitHub Actions workflow for automatic indexing & Pages deployment
├── AGENTS.md                    # This agent reference document
└── .agents/rules/               # Workspace agent rules
```

---

## 3. Core Architectural & Development Rules

### A. Client-Side RAG & Gemini API Handling (`app.js`)
* **API Key Security:** Never hardcode API keys in `app.js` or `index.html`. API keys are supplied by the user and stored in the user's browser `localStorage`.
* **Streaming & Error Handling:** Use streaming responses (`streamGenerateContent`) where possible for instant time-to-first-token. Always handle quota exhaustion (HTTP 429), model rate limits, and network disconnects with clear user notifications.
* **Exact Citation Grounding:** Every RAG response must ground and display source citations (Document Name, Clause / Section ID, Page Number, and excerpt card).

### B. Indexing Pipeline (`scripts/build_index.py`)
* **Clause Preservation:** Maintain regex patterns that identify engineering standard clauses (e.g., `ASME B31.3 Para 304.1.2`, `API 510 Sec 6.4`, `NFPA 70 Art 250`) so chunks retain precise technical metadata.
* **Batch Embedding:** Use batched Gemini embedding calls (`embedContent` / batch API) to minimize network overhead and avoid hitting rate limits during CI indexing.

### C. UI & Mobile Usability (`styles.css`, `index.html`)
* Maintain responsive, clean, high-contrast engineering UI with expandable citation cards, search filters, and full keyboard accessibility.

---

## 4. Verification Checklist

Before finalizing changes:
1. **Frontend Integrity:** Verify `index.html`, `app.js`, and `styles.css` load cleanly without runtime console errors.
2. **Script Syntax:** Run `python -m py_compile scripts/build_index.py` (and any related Python script) to ensure clean syntax.
