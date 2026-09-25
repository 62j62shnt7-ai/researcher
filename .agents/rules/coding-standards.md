---
description: "Coding standards and rules for Researcher AI"
globs: "**/*"
---

# Researcher AI — Development Rules

1. **Client API Key Security**:
   - Never embed Gemini API keys in client-side code (`app.js`, `index.html`) or commit them to the repository.

2. **Source Grounding**:
   - All AI answers must provide citations to specific document names, clauses/sections, and page numbers.

3. **Hybrid Search Pipeline**:
   - Ensure BM25 exact clause keyword matching and cosine vector similarity remain synced and balanced.
