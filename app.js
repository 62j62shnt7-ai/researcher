/**
 * Researcher AI - Client App Engine
 * Hybrid RAG (BM25 + Gemini Embeddings) & Client-side PDF/DOCX Parser
 */

(function () {
  'use strict';

  // Global State
  const state = {
    apiKey: localStorage.getItem('gemini_api_key') || '',
    model: localStorage.getItem('gemini_model') || 'gemini-1.5-flash',
    repoKB: { documents: [], chunks: [] },
    localDocs: [],
    localChunks: [],
    activeFilter: 'all',
    db: null
  };

  // DOM Elements
  const elements = {
    apiKeyInput: document.getElementById('api-key-input'),
    btnSaveKey: document.getElementById('btn-save-key'),
    btnTestKey: document.getElementById('btn-test-key'),
    btnSettings: document.getElementById('btn-settings'),
    settingsModal: document.getElementById('settings-modal'),
    settingsStatus: document.getElementById('settings-status'),
    btnUpload: document.getElementById('btn-upload'),
    fileInput: document.getElementById('file-input'),
    btnLibrary: document.getElementById('btn-library'),
    libraryModal: document.getElementById('library-modal'),
    libCount: document.getElementById('lib-count'),
    repoDocsList: document.getElementById('repo-docs-list'),
    localDocsList: document.getElementById('local-docs-list'),
    btnExportDb: document.getElementById('btn-export-db'),
    btnImportDb: document.getElementById('btn-import-db'),
    importDbInput: document.getElementById('import-db-input'),
    searchInput: document.getElementById('search-input'),
    searchClearBtn: document.getElementById('search-clear-btn'),
    searchResultsSection: document.getElementById('search-results-section'),
    resultsList: document.getElementById('results-list'),
    resultsCount: document.getElementById('results-count'),
    closeResultsBtn: document.getElementById('close-results-btn'),
    chatMessages: document.getElementById('chat-messages'),
    chatForm: document.getElementById('chat-form'),
    chatInput: document.getElementById('chat-input'),
    btnSend: document.getElementById('btn-send'),
    useRagToggle: document.getElementById('use-rag-toggle'),
    modelSelect: document.getElementById('model-select'),
    filterPills: document.querySelectorAll('.pill')
  };

  // Initialize PDF.js worker
  if (window.pdfjsLib) {
    try {
      pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
    } catch (e) {
      console.warn('PDF.js worker setup error:', e);
    }
  }

  // --- Dynamic Model Discovery ---
  async function discoverUserModels(apiKey) {
    if (!apiKey) return;
    try {
      const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        const models = data.models || [];
        const generateModels = models.filter(m => m.supportedGenerationMethods?.includes('generateContent'));
        if (generateModels.length > 0 && elements.modelSelect) {
          elements.modelSelect.innerHTML = '';
          generateModels.forEach(m => {
            const modelId = m.name.replace('models/', '');
            const opt = document.createElement('option');
            opt.value = modelId;
            opt.textContent = `${m.displayName || modelId}`;
            if (modelId === state.model || (modelId.includes('1.5-flash') && !elements.modelSelect.value)) {
              opt.selected = true;
            }
            elements.modelSelect.appendChild(opt);
          });
        }
      }
    } catch (e) {
      console.warn('Could not auto-discover models:', e);
    }
  }

  // --- IndexedDB Local Storage ---
  function initIndexedDB() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open('ResearcherLocalDB', 1);
      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains('documents')) {
          db.createObjectStore('documents', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('chunks')) {
          db.createObjectStore('chunks', { keyPath: 'id' });
        }
      };
      req.onsuccess = (e) => {
        state.db = e.target.result;
        loadLocalData().then(resolve);
      };
      req.onerror = (e) => reject(e);
    });
  }

  function loadLocalData() {
    return new Promise((resolve) => {
      if (!state.db) return resolve();
      const tx = state.db.transaction(['documents', 'chunks'], 'readonly');
      const docStore = tx.objectStore('documents');
      const chunkStore = tx.objectStore('chunks');

      const docsReq = docStore.getAll();
      const chunksReq = chunkStore.getAll();

      tx.oncomplete = () => {
        state.localDocs = docsReq.result || [];
        state.localChunks = chunksReq.result || [];
        updateLibraryUI();
        resolve();
      };
    });
  }

  function saveLocalDocument(docMeta, chunks) {
    return new Promise((resolve, reject) => {
      if (!state.db) return reject('DB not initialized');
      const tx = state.db.transaction(['documents', 'chunks'], 'readwrite');
      const docStore = tx.objectStore('documents');
      const chunkStore = tx.objectStore('chunks');

      docStore.put(docMeta);
      chunks.forEach((c) => chunkStore.put(c));

      tx.oncomplete = () => {
        loadLocalData().then(resolve);
      };
      tx.onerror = (e) => reject(e);
    });
  }

  function deleteLocalDocument(docId) {
    return new Promise((resolve) => {
      if (!state.db) return resolve();
      const tx = state.db.transaction(['documents', 'chunks'], 'readwrite');
      const docStore = tx.objectStore('documents');
      const chunkStore = tx.objectStore('chunks');

      docStore.delete(docId);
      const chunksReq = chunkStore.getAll();
      chunksReq.onsuccess = () => {
        const all = chunksReq.result || [];
        all.forEach((c) => {
          if (c.docId === docId || c.file === docId) {
            chunkStore.delete(c.id);
          }
        });
      };

      tx.oncomplete = () => {
        loadLocalData().then(resolve);
      };
    });
  }

  // --- Fetch Pre-indexed Repo Knowledge Base ---
  async function fetchRepoKB() {
    try {
      const res = await fetch('knowledge_base.json');
      if (res.ok) {
        state.repoKB = await res.json();
        console.log('Loaded repo knowledge base:', state.repoKB);
        updateLibraryUI();
      }
    } catch (e) {
      console.warn('Could not load knowledge_base.json (it will be built by GitHub Actions)', e);
    }
  }

  // --- Tokenizer & BM25 Scoring ---
  function tokenize(text) {
    return (text || '').toLowerCase().match(/\b\w+\b/g) || [];
  }

  function scoreBM25(queryTokens, chunkTokens) {
    if (!queryTokens.length || !chunkTokens.length) return 0;
    const chunkTokenMap = {};
    chunkTokens.forEach((t) => (chunkTokenMap[t] = (chunkTokenMap[t] || 0) + 1));
    let score = 0;
    queryTokens.forEach((qt) => {
      if (chunkTokenMap[qt]) {
        score += chunkTokenMap[qt] * 1.5;
      }
    });
    return score;
  }

  // --- Vector Cosine Similarity ---
  function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length || vecA.length === 0) return 0;
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < vecA.length; i++) {
      dot += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-9);
  }

  // --- Gemini API Embeddings & Chat ---
  async function fetchQueryEmbedding(query) {
    if (!state.apiKey) return null;
    const models = ['text-embedding-004', 'embedding-001'];

    for (const model of models) {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:embedContent?key=${state.apiKey}`;
      try {
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            content: { parts: [{ text: query.substring(0, 2000) }] }
          })
        });
        if (res.ok) {
          const data = await res.json();
          const vals = data.embedding?.values;
          if (vals && vals.length > 0) {
            return vals;
          }
        }
      } catch (e) {
        console.warn(`Embedding failed for ${model}:`, e);
      }
    }
    return null;
  }

  async function callGeminiChat(userPrompt, retrievedChunks) {
    if (!state.apiKey) {
      throw new Error('Please set your Gemini API key in Settings first.');
    }

    const selected = elements.modelSelect ? elements.modelSelect.value : state.model;
    const modelsToTry = [selected, 'gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-2.0-flash', 'gemini-1.5-pro'].filter(Boolean);
    let lastError = null;

    let contextText = '';
    if (retrievedChunks && retrievedChunks.length > 0) {
      contextText = 'ENGINEERING CODE & STANDARD EXCERPTS:\n\n' +
        retrievedChunks.map((c, i) =>
          `[Source ${i + 1}]: File: "${c.file}", Clause/Section: "${c.clause}", Page: ${c.page}\nExcerpt:\n${c.text}`
        ).join('\n\n---\n\n');
    }

    const systemInstruction = `You are Researcher AI, an expert engineering codes & standards assistant. 
Your goal is to provide precise, technical, and accurate answers based on engineering codes, standards, and datasheets.
If context excerpts are provided above, use them strictly to answer the user's question. 
Always cite exact files, clauses (e.g. Para 304.1.2), and page numbers in brackets like [Source 1] or [File, Page X].
If the context does not contain enough information, state what is known and clarify any limits.`;

    const contents = [
      {
        role: 'user',
        parts: [
          { text: contextText ? `${contextText}\n\nUSER QUESTION: ${userPrompt}` : userPrompt }
        ]
      }
    ];

    const body = {
      contents,
      systemInstruction: { parts: [{ text: systemInstruction }] },
      generationConfig: { temperature: 0.2, maxOutputTokens: 2048 }
    };

    for (const modelName of modelsToTry) {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${state.apiKey}`;
      try {
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });

        if (res.ok) {
          const data = await res.json();
          const candidate = data.candidates?.[0];
          const answer = candidate?.content?.parts?.map(p => p.text).join('') || 'No response generated.';
          return answer;
        } else {
          const errJson = await res.json().catch(() => ({}));
          lastError = new Error(errJson.error?.message || `HTTP ${res.status}`);
        }
      } catch (e) {
        lastError = e;
      }
    }

    throw lastError || new Error('Failed to communicate with Gemini API.');
  }

  // --- Hybrid Retriever ---
  async function performHybridSearch(query, topK = 5) {
    const qTokens = tokenize(query);
    const queryVec = await fetchQueryEmbedding(query);

    let allChunks = [];
    if (state.activeFilter === 'all' || state.activeFilter === 'repo') {
      allChunks = allChunks.concat(state.repoKB.chunks || []);
    }
    if (state.activeFilter === 'all' || state.activeFilter === 'local') {
      allChunks = allChunks.concat(state.localChunks || []);
    }

    const scored = allChunks.map((chunk) => {
      const bm25 = scoreBM25(qTokens, chunk.tokens || tokenize(chunk.text));
      let vecScore = 0;
      if (queryVec && chunk.embedding) {
        vecScore = cosineSimilarity(queryVec, chunk.embedding);
      }
      const finalScore = queryVec && chunk.embedding ? (0.3 * bm25 + 0.7 * vecScore) : bm25;
      return { chunk, score: finalScore, bm25, vecScore };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).filter(s => s.score > 0.05).map(s => ({ ...s.chunk, score: s.score }));
  }

  // --- Client-Side File Processing (PDF / DOCX / TXT) ---
  async function processUploadedFile(file) {
    const filename = file.name;
    const ext = filename.split('.').pop().toLowerCase();
    let pages = [];

    try {
      if (ext === 'pdf') {
        pages = await parsePdfFile(file);
      } else if (ext === 'docx') {
        pages = await parseDocxFile(file);
      } else {
        const text = await file.text();
        pages = [{ page: 1, text }];
      }
    } catch (e) {
      console.warn(`Direct parser failed for ${filename}, falling back to plain text reader:`, e);
      try {
        const text = await file.text();
        pages = [{ page: 1, text }];
      } catch (err) {
        throw new Error(`Could not parse ${filename}: ${err.message}`);
      }
    }

    const docId = `local_${Date.now()}_${filename.replace(/[^a-zA-Z0-9._-]/g, '_')}`;
    const chunks = [];
    let chunkIdCounter = 0;

    for (const pageInfo of pages) {
      const pageNum = pageInfo.page;
      const text = pageInfo.text || '';
      if (!text.trim()) continue;

      const chunkSize = 800;
      const overlap = 100;
      let start = 0;

      while (start < text.length) {
        const chunkStr = text.slice(start, start + chunkSize).trim();
        if (chunkStr) {
          const clauseMatch = chunkStr.match(/(?:(?:para|section|clause|article|part)\s+[\d\.]+|[\d]+\.[\d]+(?:\.[\d]+)?)/i);
          const clause = clauseMatch ? clauseMatch[0] : `Page ${pageNum}`;

          chunks.push({
            id: `${docId}_c${chunkIdCounter++}`,
            docId: docId,
            file: filename,
            page: pageNum,
            clause,
            text: chunkStr,
            tokens: tokenize(chunkStr),
            embedding: null
          });
        }
        start += (chunkSize - overlap);
      }
    }

    const docMeta = {
      id: docId,
      filename,
      pageCount: pages.length,
      chunkCount: chunks.length,
      uploadedAt: new Date().toISOString()
    };

    await saveLocalDocument(docMeta, chunks);
    return docMeta;
  }

  async function parsePdfFile(file) {
    if (!window.pdfjsLib) {
      const text = await file.text();
      return [{ page: 1, text }];
    }
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    const pages = [];

    for (let i = 1; i <= pdf.numPages; i++) {
      try {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        const text = content.items.map((item) => item.str).join(' ');
        pages.push({ page: i, text: text || '' });
      } catch (err) {
        console.warn(`Error reading page ${i} of PDF:`, err);
      }
    }
    return pages.length > 0 ? pages : [{ page: 1, text: 'PDF content extracted' }];
  }

  async function parseDocxFile(file) {
    if (!window.mammoth) {
      const text = await file.text();
      return [{ page: 1, text }];
    }
    const arrayBuffer = await file.arrayBuffer();
    const result = await mammoth.extractRawText({ arrayBuffer });
    const fullText = result.value || '';
    const chunkSize = 1000;
    const pages = [];
    let p = 1;
    for (let i = 0; i < fullText.length; i += chunkSize) {
      pages.push({ page: p++, text: fullText.slice(i, i + chunkSize) });
    }
    return pages;
  }

  // --- UI Renderers & Handlers ---
  function updateLibraryUI() {
    const repoDocs = state.repoKB.documents || [];
    const totalCount = repoDocs.length + state.localDocs.length;
    elements.libCount.textContent = totalCount;

    if (repoDocs.length === 0) {
      elements.repoDocsList.innerHTML = '<p class="doc-meta">No pre-indexed standards in repo yet.</p>';
    } else {
      elements.repoDocsList.innerHTML = repoDocs.map(d => `
        <div class="doc-item">
          <div>
            <div class="doc-name">📄 ${d.filename}</div>
            <div class="doc-meta">${d.chunk_count} chunks • ${d.page_count} pages</div>
          </div>
          <span class="pill">GitHub</span>
        </div>
      `).join('');
    }

    if (state.localDocs.length === 0) {
      elements.localDocsList.innerHTML = '<p class="doc-meta">No local documents uploaded yet.</p>';
    } else {
      elements.localDocsList.innerHTML = state.localDocs.map(d => `
        <div class="doc-item">
          <div>
            <div class="doc-name">📱 ${d.filename}</div>
            <div class="doc-meta">${d.chunkCount} chunks • ${d.pageCount} pages</div>
          </div>
          <button class="btn-text-only btn-del-doc" data-id="${d.id}">🗑️</button>
        </div>
      `).join('');

      document.querySelectorAll('.btn-del-doc').forEach(btn => {
        btn.addEventListener('click', (e) => {
          const docId = e.currentTarget.getAttribute('data-id');
          deleteLocalDocument(docId);
        });
      });
    }
  }

  function renderMessage(role, text, citations = []) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role === 'user' ? 'user-message' : 'system-message'}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '⚡';

    const body = document.createElement('div');
    body.className = 'message-body';

    if (window.marked && role === 'assistant') {
      body.innerHTML = marked.parse(text);
    } else {
      body.textContent = text;
    }

    if (citations && citations.length > 0) {
      const citationsWrapper = document.createElement('div');
      citationsWrapper.className = 'citations-wrapper';
      citationsWrapper.innerHTML = `
        <div class="citations-header">📚 Source Excerpts (${citations.length})</div>
        <div class="citations-list">
          ${citations.map((c, i) => `
            <div class="excerpt-card">
              <div class="excerpt-title">[${i + 1}] ${c.file} — ${c.clause} (Page ${c.page})</div>
              <div class="excerpt-text">"${c.text.substring(0, 220)}..."</div>
            </div>
          `).join('')}
        </div>
      `;
      body.appendChild(citationsWrapper);
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(body);
    elements.chatMessages.appendChild(msgDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
  }

  // --- Event Listeners ---
  function setupEventListeners() {
    elements.btnSettings.addEventListener('click', () => {
      elements.apiKeyInput.value = state.apiKey;
      elements.settingsModal.classList.remove('hidden');
    });

    elements.settingsModal.querySelector('.modal-close').addEventListener('click', () => {
      elements.settingsModal.classList.add('hidden');
    });

    elements.btnSaveKey.addEventListener('click', async () => {
      state.apiKey = elements.apiKeyInput.value.trim();
      localStorage.setItem('gemini_api_key', state.apiKey);
      await discoverUserModels(state.apiKey);
      elements.settingsStatus.className = 'status-msg success';
      elements.settingsStatus.textContent = 'API Key saved! Discovered available models.';
      elements.settingsStatus.classList.remove('hidden');
      setTimeout(() => {
        elements.settingsModal.classList.add('hidden');
        elements.settingsStatus.classList.add('hidden');
      }, 1200);
    });

    elements.btnTestKey.addEventListener('click', async () => {
      const testKey = elements.apiKeyInput.value.trim();
      if (!testKey) {
        elements.settingsStatus.className = 'status-msg error';
        elements.settingsStatus.textContent = 'Please enter an API Key to test.';
        elements.settingsStatus.classList.remove('hidden');
        return;
      }
      elements.settingsStatus.className = 'status-msg';
      elements.settingsStatus.textContent = 'Discovering models for your account...';
      elements.settingsStatus.classList.remove('hidden');

      await discoverUserModels(testKey);

      let chatStatus = false;
      let embedStatus = false;

      const chatModels = [elements.modelSelect?.value, 'gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-2.0-flash', 'gemini-1.5-pro'].filter(Boolean);
      let workingChatModel = '';

      for (const cm of chatModels) {
        try {
          const url = `https://generativelanguage.googleapis.com/v1beta/models/${cm}:generateContent?key=${testKey}`;
          const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ contents: [{ role: 'user', parts: [{ text: 'Ping' }] }] })
          });
          if (res.ok) {
            chatStatus = true;
            workingChatModel = cm;
            break;
          }
        } catch (e) {}
      }

      const embedModels = ['text-embedding-004', 'embedding-001'];
      let workingEmbedModel = '';

      for (const em of embedModels) {
        try {
          const url = `https://generativelanguage.googleapis.com/v1beta/models/${em}:embedContent?key=${testKey}`;
          const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content: { parts: [{ text: 'Ping' }] } })
          });
          if (res.ok) {
            embedStatus = true;
            workingEmbedModel = em;
            break;
          }
        } catch (e) {}
      }

      if (chatStatus) {
        elements.settingsStatus.className = 'status-msg success';
        elements.settingsStatus.textContent = `✔️ Chat Ready (${workingChatModel}) | ${embedStatus ? `✔️ Embedder Ready (${workingEmbedModel})` : '⚠️ Keyword-Only Retrieval'}`;
      } else {
        elements.settingsStatus.className = 'status-msg error';
        elements.settingsStatus.textContent = '❌ Invalid API Key or network blocked. Please check your key at aistudio.google.com.';
      }
    });

    elements.btnLibrary.addEventListener('click', () => {
      updateLibraryUI();
      elements.libraryModal.classList.remove('hidden');
    });

    elements.libraryModal.querySelector('.modal-close').addEventListener('click', () => {
      elements.libraryModal.classList.add('hidden');
    });

    elements.btnUpload.addEventListener('click', () => {
      elements.fileInput.click();
    });

    elements.fileInput.addEventListener('change', async (e) => {
      const files = Array.from(e.target.files);
      if (!files.length) return;

      renderMessage('assistant', `Uploading and processing ${files.length} document(s)...`);
      for (const file of files) {
        try {
          const docMeta = await processUploadedFile(file);
          renderMessage('assistant', `✅ Successfully indexed **${docMeta.filename}** (${docMeta.chunkCount} chunks, ${docMeta.pageCount} pages).`);
        } catch (err) {
          renderMessage('assistant', `❌ Error indexing ${file.name}: ${err.message}`);
        }
      }
      updateLibraryUI();
      elements.fileInput.value = '';
    });

    elements.filterPills.forEach((pill) => {
      pill.addEventListener('click', (e) => {
        elements.filterPills.forEach((p) => p.classList.remove('active'));
        e.currentTarget.classList.add('active');
        state.activeFilter = e.currentTarget.getAttribute('data-filter');
      });
    });

    let searchTimeout;
    elements.searchInput.addEventListener('input', (e) => {
      const val = e.target.value.trim();
      if (val) {
        elements.searchClearBtn.classList.remove('hidden');
      } else {
        elements.searchClearBtn.classList.add('hidden');
        elements.searchResultsSection.classList.add('hidden');
        return;
      }

      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(async () => {
        const results = await performHybridSearch(val, 6);
        elements.resultsCount.textContent = results.length;
        if (results.length === 0) {
          elements.resultsList.innerHTML = '<p class="result-text">No matching code clauses found.</p>';
        } else {
          elements.resultsList.innerHTML = results.map(r => `
            <div class="result-card">
              <div class="result-meta">${r.file} — ${r.clause} (Page ${r.page})</div>
              <div class="result-text">${r.text}</div>
            </div>
          `).join('');
        }
        elements.searchResultsSection.classList.remove('hidden');
      }, 300);
    });

    elements.searchClearBtn.addEventListener('click', () => {
      elements.searchInput.value = '';
      elements.searchClearBtn.classList.add('hidden');
      elements.searchResultsSection.classList.add('hidden');
    });

    elements.closeResultsBtn.addEventListener('click', () => {
      elements.searchResultsSection.classList.add('hidden');
    });

    elements.chatForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const prompt = elements.chatInput.value.trim();
      if (!prompt) return;

      elements.chatInput.value = '';
      renderMessage('user', prompt);

      const useRag = elements.useRagToggle.checked;
      let retrieved = [];

      if (useRag) {
        retrieved = await performHybridSearch(prompt, 5);
      }

      try {
        const answer = await callGeminiChat(prompt, retrieved);
        renderMessage('assistant', answer, retrieved);
      } catch (err) {
        renderMessage('assistant', `⚠️ **Error**: ${err.message}`);
      }
    });

    elements.btnExportDb.addEventListener('click', () => {
      const backupData = {
        documents: state.localDocs,
        chunks: state.localChunks,
        exportedAt: new Date().toISOString()
      };
      const blob = new Blob([JSON.stringify(backupData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `researcher_backup_${Date.now()}.json`;
      a.click();
    });

    elements.btnImportDb.addEventListener('click', () => {
      elements.importDbInput.click();
    });

    elements.importDbInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      try {
        const text = await file.text();
        const data = JSON.parse(text);
        if (data.documents && data.chunks) {
          for (const doc) {
            const docChunks = data.chunks.filter((c) => c.docId === doc.id || c.file === doc.filename);
            await saveLocalDocument(doc, docChunks);
          }
          alert('Backup restored successfully!');
          updateLibraryUI();
        }
      } catch (err) {
        alert('Invalid backup file: ' + err.message);
      }
    });
  }

  async function init() {
    await initIndexedDB();
    await fetchRepoKB();
    if (state.apiKey) {
      await discoverUserModels(state.apiKey);
    }
    setupEventListeners();
    console.log('Researcher AI Web App initialized.');
  }

  window.addEventListener('DOMContentLoaded', init);
})();
