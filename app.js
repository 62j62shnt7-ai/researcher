/**
 * Researcher AI - Client App Engine
 * Hybrid RAG (BM25 + Gemini Embeddings) with Cloud Storage & Selective Code Picker
 */

(function () {
  'use strict';

  // Global State
  const state = {
    apiKey: localStorage.getItem('gemini_api_key') || '',
    model: localStorage.getItem('gemini_model') || 'gemini-3-flash-preview',
    discoveredModels: [],
    cloudUrl: localStorage.getItem('cloud_storage_url') || '',

    repoKB: { documents: [], chunks: [], embedding_model: '', dimensions: 0 },
    localDocs: [],
    localChunks: [],
    chatHistory: [], // Multi-turn conversational memory
    activeFilter: 'all',
    db: null,
    fetchedDriveFiles: []
  };

  // DOM Elements holder
  const elements = {};

  function initElements() {
    elements.apiKeyInput = document.getElementById('api-key-input');
    elements.chatModelSelect = document.getElementById('chat-model-select');
    elements.modelSelect = document.getElementById('model-select');
    elements.embeddingEngineSelect = document.getElementById('embedding-engine-select');
    elements.deepReasoningToggle = document.getElementById('deep-reasoning-toggle');

    elements.cloudUrlInput = document.getElementById('cloud-url-input');

    elements.btnFetchCloud = document.getElementById('btn-fetch-cloud');
    elements.driveFilesContainer = document.getElementById('drive-files-container');
    elements.driveFilesList = document.getElementById('drive-files-list');
    elements.btnSelectAllDrive = document.getElementById('btn-select-all-drive');
    elements.btnDeselectAllDrive = document.getElementById('btn-deselect-all-drive');
    elements.btnImportSelectedDrive = document.getElementById('btn-import-selected-drive');
    elements.btnSaveKey = document.getElementById('btn-save-key');
    elements.btnTestKey = document.getElementById('btn-test-key');
    elements.btnSettings = document.getElementById('btn-settings');
    elements.settingsModal = document.getElementById('settings-modal');
    elements.settingsStatus = document.getElementById('settings-status');
    elements.btnClearChat = document.getElementById('btn-clear-chat');
    elements.btnUpload = document.getElementById('btn-upload');
    elements.fileInput = document.getElementById('file-input');
    elements.btnLibrary = document.getElementById('btn-library');
    elements.libraryModal = document.getElementById('library-modal');
    elements.libCount = document.getElementById('lib-count');
    elements.repoDocsList = document.getElementById('repo-docs-list');
    elements.localDocsList = document.getElementById('local-docs-list');
    elements.btnExportDb = document.getElementById('btn-export-db');
    elements.btnImportDb = document.getElementById('btn-import-db');
    elements.importDbInput = document.getElementById('import-db-input');
    elements.searchInput = document.getElementById('search-input');
    elements.searchClearBtn = document.getElementById('search-clear-btn');
    elements.searchResultsSection = document.getElementById('search-results-section');
    elements.resultsList = document.getElementById('results-list');
    elements.resultsCount = document.getElementById('results-count');
    elements.closeResultsBtn = document.getElementById('close-results-btn');
    elements.chatMessages = document.getElementById('chat-messages');
    elements.chatForm = document.getElementById('chat-form');
    elements.chatInput = document.getElementById('chat-input');
    elements.btnSend = document.getElementById('btn-send');
    elements.useRagToggle = document.getElementById('use-rag-toggle');
    elements.modelSelect = document.getElementById('model-select');
    elements.docScopeSelect = document.getElementById('doc-scope-select');
    elements.filterPills = document.querySelectorAll('.pill');
  }

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
    if (!apiKey) return [];
    try {
      const url = `https://generativelanguage.googleapis.com/v1beta/models`;
      let res = await fetch(url, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey }
      }).catch(() => null);

      let models = [];
      if (res && res.ok) {
        const data = await res.json();
        models = data.models || [];
      } else {
        res = await fetch(`${url}?key=${apiKey}`).catch(() => null);
        if (res && res.ok) {
          const data = await res.json();
          models = data.models || [];
        }
      }

      const generateModels = models.filter(m => m.supportedGenerationMethods?.includes('generateContent'));
      // Sort to prioritize latest models (3-flash-preview, 3.8-flash, 3.7-flash, 3.5-flash, flash-latest, gemma-4)
      generateModels.sort((a, b) => {
        const getPriority = (name) => {
          if (name.includes('3-flash-preview')) return 1;
          if (name.includes('3.8-flash')) return 2;
          if (name.includes('3.7-flash')) return 3;
          if (name.includes('3.5-flash')) return 4;
          if (name.includes('flash-latest')) return 5;
          if (name.includes('gemma-4')) return 6;
          if (name.includes('pro')) return 8;
          return 10;
        };
        return getPriority(a.name) - getPriority(b.name);
      });
      const generateModelNames = generateModels.map(m => m.name.replace('models/', ''));
      state.discoveredModels = generateModelNames;

      const selectElements = [
        elements.chatModelSelect,
        elements.modelSelect,
        document.getElementById('chat-model-select'),
        document.getElementById('model-select')
      ].filter((el, idx, self) => el && self.indexOf(el) === idx);

      if (generateModels.length > 0 && selectElements.length > 0) {
        if (!generateModelNames.includes(state.model)) {
          state.model = generateModelNames[0];
          localStorage.setItem('gemini_model', state.model);
        }

        selectElements.forEach(selectEl => {
          selectEl.innerHTML = '';
          generateModels.forEach(m => {
            const modelId = m.name.replace('models/', '');
            const opt = document.createElement('option');
            opt.value = modelId;
            opt.textContent = `${m.displayName || modelId}`;
            if (modelId === state.model) {
              opt.selected = true;
            }
            selectEl.appendChild(opt);
          });
        });
      } else if (selectElements.length > 0) {
        selectElements.forEach(selectEl => {
          selectEl.innerHTML = '<option value="">❌ No generateContent models found for this key</option>';
        });
      }
      return generateModelNames;
    } catch (e) {
      console.warn('Could not auto-discover models:', e);
      return [];
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
        populateDocScopeSelect();
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

  function clearAllLocalDocuments() {
    return new Promise((resolve) => {
      if (!state.db) return resolve();
      const tx = state.db.transaction(['documents', 'chunks'], 'readwrite');
      tx.objectStore('documents').clear();
      tx.objectStore('chunks').clear();
      tx.oncomplete = () => {
        loadLocalData().then(resolve);
      };
    });
  }


  // --- Fetch Pre-indexed Knowledge Base (Cloud or Repo) ---
  async function fetchRepoKB() {
    const targetUrl = state.cloudUrl || 'knowledge_base.json';
    try {
      if (targetUrl.includes('drive.google.com') || targetUrl.includes('dropbox.com') || targetUrl.includes('onedrive.live.com') || targetUrl.includes('1drv.ms') || targetUrl.match(/\/folders\//)) {
        return;
      }

      const res = await fetch(targetUrl);
      if (res.ok) {
        state.repoKB = await res.json();
        console.log('Loaded knowledge base from:', targetUrl, state.repoKB);
        updateLibraryUI();
        populateDocScopeSelect();
      }
    } catch (e) {
      console.warn('Could not load knowledge base from:', targetUrl, e);
    }
  }

  // --- Fetch Direct Single PDF Link ---
  async function fetchDirectPdfUrl(urlInput) {
    if (!urlInput) return false;
    const updateStatus = (msg, isErr = false) => {
      if (elements.settingsStatus) {
        elements.settingsStatus.className = isErr ? 'status-msg error' : 'status-msg';
        elements.settingsStatus.textContent = msg;
        elements.settingsStatus.classList.remove('hidden');
      }
    };

    let downloadUrl = urlInput.trim();
    let filename = 'Cloud_Document.pdf';
    const driveMatch = urlInput.match(/\/file\/d\/([a-zA-Z0-9_-]+)/) || urlInput.match(/id=([a-zA-Z0-9_-]+)/);

    if (urlInput.toLowerCase().includes('dropbox.com')) {
      downloadUrl = urlInput.replace('www.dropbox.com', 'dl.dropboxusercontent.com').replace('?dl=0', '?raw=1');
      if (!downloadUrl.includes('raw=1') && !downloadUrl.includes('dl=1')) {
        downloadUrl += (downloadUrl.includes('?') ? '&raw=1' : '?raw=1');
      }
      const urlParts = urlInput.split('/');
      const lastPart = urlParts[urlParts.length - 1].split('?')[0];
      if (lastPart && lastPart.toLowerCase().endsWith('.pdf')) {
        filename = decodeURIComponent(lastPart);
      }
    } else if (urlInput.toLowerCase().includes('onedrive.live.com') || urlInput.toLowerCase().includes('1drv.ms')) {
      downloadUrl = urlInput.replace('/redir?', '/download?').replace('/embed?', '/download?');
    } else if (driveMatch) {
      const fileId = driveMatch[1];
      downloadUrl = `https://drive.usercontent.google.com/download?id=${fileId}&export=download`;
      filename = `Google_Drive_Code_${fileId.substring(0, 6)}.pdf`;
    }

    updateStatus(`📥 Downloading & parsing single PDF code...`);

    const fetchUrls = [
      downloadUrl,
      `https://corsproxy.io/?${encodeURIComponent(downloadUrl)}`,
      `https://api.allorigins.win/raw?url=${encodeURIComponent(downloadUrl)}`
    ];

    for (const pUrl of fetchUrls) {
      try {
        const res = await fetch(pUrl);
        if (res.ok) {
          const blob = await res.blob();
          if (blob.size > 500) {
            const fileObj = new File([blob], filename, { type: 'application/pdf' });
            const docMeta = await processUploadedFile(fileObj);
            updateLibraryUI();
            populateDocScopeSelect();
            updateStatus(`✅ Successfully imported ${docMeta.filename} (${docMeta.pageCount} pages, ${docMeta.chunkCount} chunks)!`);
            return true;
          }
        }
      } catch (e) {
        console.warn('Direct fetch failed:', e);
      }
    }

    updateStatus(`❌ Could not fetch single PDF link. Use "Upload Code" button to pick PDFs directly from your device.`, true);
    return false;
  }

  // --- Scan Folder Contents & Present Interactive Checklist ---
  async function scanGoogleDriveFolder(urlOrId) {
    if (!urlOrId) return [];
    const updateStatus = (msg, isErr = false) => {
      if (elements.settingsStatus) {
        elements.settingsStatus.className = isErr ? 'status-msg error' : 'status-msg';
        elements.settingsStatus.textContent = msg;
        elements.settingsStatus.classList.remove('hidden');
      }
    };

    if (urlOrId.includes('/file/d/') || urlOrId.endsWith('.pdf')) {
      return await fetchDirectPdfUrl(urlOrId);
    }

    updateStatus('🔄 Scanning cloud folder contents...');
    const fileMap = new Map();

    const folderMatches = Array.from(urlOrId.matchAll(/\/folders\/([a-zA-Z0-9_-]{25,50})/g));
    let folderIds = folderMatches.map(m => m[1]);

    if (folderIds.length === 0) {
      const rawIdMatch = urlOrId.trim().match(/^[a-zA-Z0-9_-]{25,50}$/);
      if (rawIdMatch) folderIds = [rawIdMatch[0]];
    }

    for (const folderId of folderIds) {
      let htmlText = '';
      const proxyUrls = [
        `https://corsproxy.io/?https://drive.google.com/drive/folders/${folderId}`,
        `https://api.allorigins.win/raw?url=${encodeURIComponent('https://drive.google.com/drive/folders/' + folderId)}`,
        `https://corsproxy.io/?https://drive.google.com/embeddedfolderview?id=${folderId}`
      ];

      for (const pUrl of proxyUrls) {
        try {
          const res = await fetch(pUrl);
          if (res.ok) {
            const txt = await res.text();
            if (txt && txt.length > 300) {
              htmlText += '\n' + txt;
            }
          }
        } catch (e) {}
      }

      const jsonMatches = Array.from(htmlText.matchAll(/\["([a-zA-Z0-9_-]{25,50})",\s*"([^"\\]+\.(?:pdf|docx|txt|csv|xlsx|md))"/gi));
      jsonMatches.forEach(m => {
        if (m[1] && m[2] && !m[2].includes('<') && !m[2].includes('>')) {
          fileMap.set(m[1], m[2]);
        }
      });
    }

    const fileList = Array.from(fileMap.entries()).map(([id, name]) => ({ id, name }));

    if (fileList.length === 0) {
      return await fetchDirectPdfUrl(urlOrId);
    }

    state.fetchedDriveFiles = fileList;
    renderDriveChecklist(fileList);
    updateStatus(`📋 Found ${fileList.length} code(s). Select the specific PDFs you want to parse below.`);
    return fileList;
  }

  function renderDriveChecklist(files) {
    if (!elements.driveFilesList || !elements.driveFilesContainer) return;
    elements.driveFilesList.innerHTML = files.map(f => `
      <label class="doc-item" style="cursor: pointer; display: flex; gap: 0.5rem; align-items: center;">
        <input type="checkbox" class="drive-file-cb" data-id="${f.id}" data-name="${f.name}" checked>
        <span class="doc-name">📄 ${f.name}</span>
      </label>
    `).join('');
    elements.driveFilesContainer.classList.remove('hidden');
  }

  async function importSelectedDriveFiles() {
    const cbs = document.querySelectorAll('.drive-file-cb:checked');
    if (!cbs || cbs.length === 0) {
      alert('Please select at least one code to import.');
      return;
    }

    const selectedFiles = Array.from(cbs).map(cb => ({
      id: cb.getAttribute('data-id'),
      name: cb.getAttribute('data-name')
    }));

    const updateStatus = (msg) => {
      if (elements.settingsStatus) {
        elements.settingsStatus.className = 'status-msg';
        elements.settingsStatus.textContent = msg;
        elements.settingsStatus.classList.remove('hidden');
      }
    };

    updateStatus(`📥 Downloading & parsing ${selectedFiles.length} selected document(s)...`);
    let successCount = 0;

    for (let i = 0; i < selectedFiles.length; i++) {
      const item = selectedFiles[i];
      updateStatus(`📥 Parsing (${i + 1}/${selectedFiles.length}): ${item.name}...`);
      const downloadUrls = [
        `https://drive.usercontent.google.com/download?id=${item.id}&export=download`,
        `https://corsproxy.io/?https://drive.google.com/uc?export=download&id=${item.id}`,
        `https://api.allorigins.win/raw?url=${encodeURIComponent('https://drive.google.com/uc?export=download&id=' + item.id)}`
      ];

      for (const dUrl of downloadUrls) {
        try {
          const fileRes = await fetch(dUrl);
          if (fileRes.ok) {
            const blob = await fileRes.blob();
            if (blob.size > 500) {
              const fileObj = new File([blob], item.name, { type: 'application/pdf' });
              await processUploadedFile(fileObj);
              successCount++;
              break;
            }
          }
        } catch (err) {}
      }
    }

    updateLibraryUI();
    populateDocScopeSelect();
    updateStatus(`✅ Successfully imported ${successCount} selected code(s)!`);
  }

  // --- Populate Code Scope Selector Dropdown ---
  function populateDocScopeSelect() {
    if (!elements.docScopeSelect) return;
    const currentScope = elements.docScopeSelect.value || 'all';
    elements.docScopeSelect.innerHTML = '<option value="all">🌐 All Codes & Standards</option>';

    const repoDocs = state.repoKB.documents || [];
    repoDocs.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.filename;
      opt.textContent = `📄 ${d.filename}`;
      if (d.filename === currentScope) opt.selected = true;
      elements.docScopeSelect.appendChild(opt);
    });

    state.localDocs.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.filename;
      opt.textContent = `📱 ${d.filename}`;
      if (d.filename === currentScope) opt.selected = true;
      elements.docScopeSelect.appendChild(opt);
    });
  }

  // --- Tokenizer & BM25 Scoring ---
  function tokenize(text) {
    if (!text) return [];
    // Extract standard technical clauses and tokens, preserving dots, hyphens, and colons (e.g. 304.1.2, UG-27, ISO 9001:2015)
    return text.toLowerCase().match(/[a-z0-9]+(?:[\.\-_:][a-z0-9]+)*/gi) || [];
  }

  function scoreBM25(queryTokens, chunkTokens) {
    if (!queryTokens.length || !chunkTokens.length) return 0;
    const chunkTokenMap = {};
    chunkTokens.forEach((t) => (chunkTokenMap[t] = (chunkTokenMap[t] || 0) + 1));
    let score = 0;
    queryTokens.forEach((qt) => {
      if (chunkTokenMap[qt]) {
        // Boost technical clause tokens with numbers, dots or dashes
        const weight = qt.includes('.') || qt.includes('-') || qt.length > 5 ? 3.0 : 1.5;
        score += chunkTokenMap[qt] * weight;
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

  let localBrowserEmbedder = null;
  async function getLocalBrowserEmbedder() {
    if (localBrowserEmbedder) return localBrowserEmbedder;
    if (window.transformers && window.transformers.pipeline) {
      try {
        console.log('Loading free client-side browser embedding model (Xenova/bge-small-en-v1.5)...');
        localBrowserEmbedder = await window.transformers.pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5');
        return localBrowserEmbedder;
      } catch (e) {
        console.warn('Transformers.js client pipeline error:', e);
      }
    }
    return null;
  }

  // --- Embeddings & Chat ---
  async function fetchQueryEmbedding(query) {
    // 1. Try free Transformers.js browser embedding
    const localEmbedder = await getLocalBrowserEmbedder();
    if (localEmbedder) {
      try {
        const out = await localEmbedder(query, { pooling: 'mean', normalize: true });
        return Array.from(out.data);
      } catch (e) {
        console.warn('Client browser embedding computation failed:', e);
      }
    }

    // 2. Fallback to Gemini API if key is present
    if (state.apiKey) {
      const models = ['gemini-embedding-2', 'text-embedding-004'];

      for (const model of models) {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:embedContent`;
        try {
          const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'x-goog-api-key': state.apiKey },
            body: JSON.stringify({
              model: `models/${model}`,
              content: { parts: [{ text: query.substring(0, 8000) }] }
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
    }

    return null;
  }


  async function streamGeminiChat(userPrompt, retrievedChunks, docScope = 'all', onDelta = null) {
    if (!state.apiKey) {
      throw new Error('Please set your Gemini API key in Settings first.');
    }

    const selectedModel = state.model || (elements.chatModelSelect && elements.chatModelSelect.value) || (elements.modelSelect && elements.modelSelect.value) || 'gemini-3-flash-preview';
    const fallbackList = ['gemini-3-flash-preview', 'gemini-3.7-flash', 'gemini-3.8-flash', 'gemma-4-31b-it'];
    const modelsToTry = [selectedModel, ...fallbackList].filter((m, i, arr) => m && arr.indexOf(m) === i && !m.includes('2.5'));

    let lastError = null;

    let srcText = '';
    if (retrievedChunks && retrievedChunks.length > 0) {
      srcText = retrievedChunks.map((c, i) => {
        const loc = c.page ? `Page ${c.page}${c.clause ? ', ' + c.clause : ''}` : (c.clause || '');
        return `[${i + 1}] (${c.file}${loc ? ', ' + loc : ''})\n${c.text}`;
      }).join('\n\n---\n\n');
    } else {
      srcText = '(no matching excerpts found in the library)';
    }

    const scopeNote = (docScope && docScope !== 'all') ? `\n\nFocus specifically on the document: "${docScope}".` : '';

    const systemInstruction = `You are an expert engineering assistant with deep knowledge of codes, standards, and engineering practice. You answer questions using excerpts retrieved from the user's document library, shown in SOURCES below.${scopeNote}

How to answer:
1. Start with a direct answer to the question.
2. Then EXPLAIN it properly: what it means in practice, why the requirement exists where evident, what conditions/exceptions apply, and how the pieces relate. Do not just quote fragments back — interpret and synthesize them like a senior engineer explaining to a colleague.
3. Combine information across multiple sources when they cover the same topic; point out when sources differ or when a requirement in one place is modified by another.
4. Quote exact clause numbers, values, formulas, tolerances, and table data when present. Never invent clause numbers or values.
5. Cite sources inline with bracketed numbers, e.g. [1] or [2][3], so the user can verify.
6. If the sources only partially answer the question, answer what you can from them, then clearly separate any additional general engineering knowledge with "(general knowledge, not from your documents)". If the sources contain nothing relevant, say so plainly.
7. Use markdown (headings, tables, lists) when it makes the answer clearer.

SOURCES:
${srcText}`;

    // Multi-turn conversation contents:
    const contents = [];
    const recentTurns = (state.chatHistory || []).slice(-8);
    for (const turn of recentTurns) {
      contents.push({
        role: turn.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: turn.content }]
      });
    }

    // Current user prompt
    contents.push({
      role: 'user',
      parts: [{ text: userPrompt }]
    });

    const isDeepReasoning = elements.deepReasoningToggle ? elements.deepReasoningToggle.checked : true;

    for (const modelName of modelsToTry) {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:streamGenerateContent?alt=sse&key=${state.apiKey}`;

      const generationConfig = {
        temperature: 0.2,
        maxOutputTokens: 8192
      };

      const body = {
        contents,
        systemInstruction: { parts: [{ text: systemInstruction }] },
        generationConfig
      };

      try {
        let res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'x-goog-api-key': state.apiKey },
          body: JSON.stringify(body)
        });

        // If thinkingConfig causes 400 Bad Request on an unsupported model, retry without it
        if (res.status === 400 && generationConfig.thinkingConfig) {
          delete body.generationConfig.thinkingConfig;
          res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'x-goog-api-key': state.apiKey },
            body: JSON.stringify(body)
          });
        }

        if (!res.ok) {
          const errJson = await res.json().catch(() => ({}));
          lastError = new Error(errJson.error?.message || `HTTP ${res.status}`);
          console.warn(`Model ${modelName} stream call failed:`, lastError);
          continue;
        }

        let fullText = '';
        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop(); // keep remainder

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const jsonStr = line.slice(6).trim();
              if (!jsonStr) continue;
              try {
                const data = JSON.parse(jsonStr);
                const parts = data.candidates?.[0]?.content?.parts || [];
                for (const part of parts) {
                  if (part.thought) continue;
                  let t = part.text || '';
                  t = t.replace(/<think>[\s\S]*?(<\/think>|$)/g, '').replace(/<thought>[\s\S]*?(<\/thought>|$)/g, '');
                  if (t) {
                    fullText += t;
                    if (onDelta) onDelta(t, fullText);
                  }
                }
              } catch (parseErr) {}
            }
          }
        }

        if (fullText) {
          if (modelName !== selectedModel) {
            fullText += `\n\n*(Note: Requested model \`${selectedModel}\` was unavailable; response generated via fallback \`${modelName}\`)*`;
          }
          return { answer: fullText, modelUsed: modelName };
        }
      } catch (e) {
        lastError = e;
      }
    }

    throw lastError || new Error('Failed to communicate with Gemini API.');
  }


  // --- Continuous Context Neighbor Expansion (matching offline app rag.py) ---
  function expandChunkWithNeighbors(chunk, allChunks, maxChars = 3500) {
    if (!chunk || !chunk.text) return '';
    let text = chunk.text;
    if (text.length >= maxChars) return text.substring(0, maxChars);

    const chunkFile = chunk.file || '';
    const m = String(chunk.id || '').match(/^(.*?)_(\d+)$/);
    const idx = m ? parseInt(m[2], 10) : (chunk.chunkIndex != null ? chunk.chunkIndex : null);
    const prefix = m ? m[1] : chunkFile;

    if (idx !== null) {
      const budget = maxChars - text.length;
      if (budget > 300) {
        const nextChunk = allChunks.find(c =>
          c.id === `${prefix}_${idx + 1}` ||
          (c.file === chunkFile && c.chunkIndex === idx + 1)
        );
        const prevChunk = allChunks.find(c =>
          c.id === `${prefix}_${idx - 1}` ||
          (c.file === chunkFile && c.chunkIndex === idx - 1)
        );
        const halfBudget = Math.floor(budget / 2);
        if (nextChunk && nextChunk.text) {
          text = text + "\n\n" + nextChunk.text.substring(0, halfBudget);
        }
        if (prevChunk && prevChunk.text) {
          text = prevChunk.text.slice(-halfBudget) + "\n\n" + text;
        }
      }
    }
    return text.substring(0, maxChars);
  }

  // --- Hybrid Retriever (RRF fused BM25 + Vector Search with Neighbor Expansion) ---
  async function performHybridSearch(query, topK = 8, docScope = 'all') {
    const qTokens = tokenize(query);
    let queryVec = null;
    try {
      queryVec = await fetchQueryEmbedding(query);
    } catch (e) {
      queryVec = null;
    }

    let allChunks = [];
    if (state.activeFilter === 'all' || state.activeFilter === 'repo') {
      allChunks = allChunks.concat(state.repoKB.chunks || []);
    }
    if (state.activeFilter === 'all' || state.activeFilter === 'local') {
      allChunks = allChunks.concat(state.localChunks || []);
    }

    if (docScope && docScope !== 'all') {
      allChunks = allChunks.filter(c => c.file === docScope || c.docId === docScope);
    }

    if (allChunks.length === 0) return [];

    // Check dimension compatibility to avoid silent 0-score bug
    const sampleChunk = allChunks.find(c => c.embedding && Array.isArray(c.embedding) && c.embedding.length > 0);
    const hasDimMismatch = (queryVec && sampleChunk && queryVec.length !== sampleChunk.embedding.length);

    // 1. BM25 keyword rankings
    const bm25List = allChunks.map(chunk => ({
      chunk,
      score: scoreBM25(qTokens, chunk.tokens || tokenize(chunk.text))
    })).filter(x => x.score > 0).sort((a, b) => b.score - a.score);

    // 2. Vector semantic rankings
    let vecList = [];
    if (queryVec && !hasDimMismatch) {
      vecList = allChunks.map(chunk => {
        let vecScore = 0;
        if (chunk.embedding && queryVec.length === chunk.embedding.length) {
          vecScore = cosineSimilarity(queryVec, chunk.embedding);
        }
        return { chunk, score: vecScore };
      }).filter(x => x.score > 0.3).sort((a, b) => b.score - a.score);
    }

    // 3. Reciprocal Rank Fusion (RRF) matching offline store.py (vector_weight = 0.6)
    const K = 60.0;
    const vectorWeight = 0.6;
    const fusedScores = new Map();
    const chunkMap = new Map();

    bm25List.slice(0, 40).forEach((item, rank) => {
      const cid = item.chunk.id || `${item.chunk.file}_${item.chunk.page}_${rank}`;
      chunkMap.set(cid, item.chunk);
      fusedScores.set(cid, (fusedScores.get(cid) || 0) + (1 - vectorWeight) / (K + rank + 1));
    });

    vecList.slice(0, 40).forEach((item, rank) => {
      const cid = item.chunk.id || `${item.chunk.file}_${item.chunk.page}_${rank}`;
      chunkMap.set(cid, item.chunk);
      fusedScores.set(cid, (fusedScores.get(cid) || 0) + vectorWeight / (K + rank + 1));
    });

    let rankedEntries = Array.from(fusedScores.entries()).sort((a, b) => b[1] - a[1]).slice(0, topK);

    let topHits = rankedEntries.map(([cid, score]) => {
      const chunk = chunkMap.get(cid);
      return {
        ...chunk,
        text: expandChunkWithNeighbors(chunk, allChunks, 3500),
        score
      };
    });

    // Fallback: if broad question yields no specific keyword hits, return leading chunks (scope, table of contents, introduction)
    if (topHits.length === 0 && allChunks.length > 0) {
      topHits = allChunks.slice(0, Math.min(topK, 8)).map((c) => ({
        ...c,
        text: expandChunkWithNeighbors(c, allChunks, 3500),
        score: 0.1
      }));
    }

    return topHits;
  }

  // --- Client-Side File Processing (PDF / DOCX / TXT) ---
  async function processUploadedFile(file) {
    const filename = file.name;
    const ext = filename.split('.').pop().toLowerCase();
    let pages = [];

    try {
      if (ext === 'pdf') {
        pages = await parsePdfFile(file);
      } else if (ext === 'docx' || ext === 'doc') {
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

      const chunkSize = 1800;
      const overlap = 200;
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

    // Save immediately into IndexedDB so the document is READY TO CHAT & SEARCH IN < 1 SECOND!
    await saveLocalDocument(docMeta, chunks);
    updateLibraryUI();
    populateDocScopeSelect();

    // Progress banner feedback
    const progressBanner = document.getElementById('embedding-progress-banner');
    const progressStatusMsg = document.getElementById('progress-status-msg');
    const progressPercent = document.getElementById('progress-percent');
    const progressBarFill = document.getElementById('progress-bar-fill');

    if (progressBanner) {
      progressBanner.classList.remove('hidden');
      if (progressStatusMsg) progressStatusMsg.textContent = `⚡ Document indexed! Ready for instant search & chat (${filename})`;
      if (progressPercent) progressPercent.textContent = `100%`;
      if (progressBarFill) progressBarFill.style.width = `100%`;
      setTimeout(() => {
        progressBanner.classList.add('hidden');
      }, 1800);
    }

    return docMeta;
  }


  async function parsePdfFile(file) {
    const arrayBuffer = await file.arrayBuffer();
    const pages = [];

    if (window.pdfjsLib) {
      try {
        const loadingTask = pdfjsLib.getDocument({
          data: arrayBuffer,
          cMapUrl: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/cmaps/',
          cMapPacked: true
        });
        const pdf = await loadingTask.promise;
        for (let i = 1; i <= pdf.numPages; i++) {
          try {
            const page = await pdf.getPage(i);
            const content = await page.getTextContent();
            const text = content.items.map(item => item.str).join(' ');
            if (text.trim()) {
              pages.push({ page: i, text: text.trim() });
            }
          } catch (err) {
            console.warn(`Page ${i} text extraction warning:`, err);
          }
        }
      } catch (e) {
        console.warn('PDF.js primary parser failed:', e);
      }
    }

    if (pages.length > 0) return pages;

    // Fallback: Direct PDF stream text decoder
    try {
      const decoder = new TextDecoder('utf-8');
      const rawText = decoder.decode(arrayBuffer);
      const textMatches = Array.from(rawText.matchAll(/\(([^()]{3,1000})\)\s*Tj/g));
      const extractedStr = textMatches.map(m => m[1]).join(' ');
      if (extractedStr.trim().length > 100) {
        return [{ page: 1, text: extractedStr }];
      }
    } catch (err) {}

    return [{ page: 1, text: 'PDF Document content extracted.' }];
  }

  async function parseDocxFile(file) {
    if (window.mammoth) {
      try {
        const arrayBuffer = await file.arrayBuffer();
        const result = await mammoth.extractRawText({ arrayBuffer });
        const fullText = (result.value || '').trim();
        if (fullText) {
          const chunkSize = 1000;
          const pages = [];
          let p = 1;
          for (let i = 0; i < fullText.length; i += chunkSize) {
            pages.push({ page: p++, text: fullText.slice(i, i + chunkSize) });
          }
          return pages;
        }
      } catch (e) {
        console.warn('Mammoth docx extraction failed, using text fallback:', e);
      }
    }
    try {
      const text = await file.text();
      const cleanText = text.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, ' ').trim();
      return [{ page: 1, text: cleanText }];
    } catch (err) {
      return [{ page: 1, text: '' }];
    }
  }

  // --- UI Renderers & Handlers ---
  function updateLibraryUI() {
    const repoDocs = state.repoKB.documents || [];
    const totalCount = repoDocs.length + state.localDocs.length;
    if (elements.libCount) elements.libCount.textContent = totalCount;

    if (elements.repoDocsList) {
      if (repoDocs.length === 0) {
        elements.repoDocsList.innerHTML = '<p class="doc-meta">No pre-indexed standards in repo or cloud yet.</p>';
      } else {
        elements.repoDocsList.innerHTML = repoDocs.map(d => `
          <div class="doc-item">
            <div>
              <div class="doc-name">📄 ${d.filename}</div>
              <div class="doc-meta">${d.chunk_count} chunks • ${d.page_count} pages</div>
            </div>
            <span class="pill">Cloud/Repo</span>
          </div>
        `).join('');
      }
    }

    if (elements.localDocsList) {
      if (state.localDocs.length === 0) {
        elements.localDocsList.innerHTML = '<p class="doc-meta">No local documents uploaded yet.</p>';
      } else {
        const totalLocalChunks = state.localDocs.reduce((acc, d) => acc + (d.chunkCount || 0), 0);
        elements.localDocsList.innerHTML = `
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
            <span class="doc-meta" style="font-weight:600;">${state.localDocs.length} local document(s) • ${totalLocalChunks} chunks</span>
            <button id="btn-clear-all-local" class="btn-text-only" style="color:var(--bad, #e0635c); font-size:0.8rem; cursor:pointer;" type="button">🗑️ Clear All</button>
          </div>
        ` + state.localDocs.map(d => `
          <div class="doc-item">
            <div>
              <div class="doc-name">📱 ${d.filename}</div>
              <div class="doc-meta">${d.chunkCount} chunks • ${d.pageCount} pages</div>
            </div>
            <button class="btn-text-only btn-del-doc" data-id="${d.id}">🗑️</button>
          </div>
        `).join('');

        const clearBtn = document.getElementById('btn-clear-all-local');
        if (clearBtn) {
          clearBtn.addEventListener('click', async () => {
            if (confirm('Are you sure you want to clear all uploaded local documents?')) {
              await clearAllLocalDocuments();
            }
          });
        }

        document.querySelectorAll('.btn-del-doc').forEach(btn => {
          btn.addEventListener('click', (e) => {
            const docId = e.currentTarget.getAttribute('data-id');
            deleteLocalDocument(docId);
          });
        });
      }
    }

  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function buildExcerptsDrawerHtml(citations) {
    if (!citations || !citations.length) return '';
    return `
      <details class="sources-drawer" open>
        <summary class="sources-summary">📄 ${citations.length} Source Excerpt${citations.length > 1 ? 's' : ''} (click to collapse)</summary>
        <div class="citations-list">
          ${citations.map((c, i) => {
            const loc = c.page ? `Page ${c.page}${c.clause ? ', ' + escapeHtml(c.clause) : ''}` : escapeHtml(c.clause || '');
            const head = `[${i + 1}] ${escapeHtml(c.file)}${loc ? ' — ' + loc : ''}`;
            return `
              <div class="excerpt-card" id="source-card-${i + 1}" data-index="${i + 1}">
                <div class="excerpt-header">
                  <span class="excerpt-title">${head}</span>
                  <button type="button" class="btn-copy-excerpt" title="Copy excerpt to clipboard">📋 Copy</button>
                </div>
                <div class="excerpt-text">${escapeHtml(c.text)}</div>
              </div>
            `;
          }).join('')}
        </div>
      </details>
    `;
  }

  function renderFormattedContent(container, markdownText) {
    if (!container) return;
    if (window.marked) {
      // Process citations into interactive badges
      let processed = (markdownText || '').replace(/\[(?:Source\s*)?(\d{1,2})\]/gi, (match, num) => {
        return `<a class="citation-pill" data-source-index="${num}" href="javascript:void(0);" title="Jump to Source [${num}]">[${num}]</a>`;
      });

      let html = marked.parse(processed);
      container.innerHTML = html;

      // Wrap tables for responsive horizontal scrolling on mobile
      container.querySelectorAll('table').forEach(tbl => {
        if (!tbl.parentElement.classList.contains('table-responsive')) {
          const wrapper = document.createElement('div');
          wrapper.className = 'table-responsive';
          tbl.parentNode.insertBefore(wrapper, tbl);
          wrapper.appendChild(tbl);
        }
      });

      // Render math formulas with KaTeX
      if (window.renderMathInElement) {
        try {
          renderMathInElement(container, {
            delimiters: [
              { left: '$$', right: '$$', display: true },
              { left: '$', right: '$', display: false },
              { left: '\\(', right: '\\)', display: false },
              { left: '\\[', right: '\\]', display: true }
            ],
            throwOnError: false
          });
        } catch (e) {
          console.warn('KaTeX math rendering warning:', e);
        }
      }
    } else {
      container.textContent = markdownText;
    }
  }

  function renderMessage(role, text, citations = [], modelBadge = '') {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role === 'user' ? 'user-message' : 'system-message'}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '⚡';

    const body = document.createElement('div');
    body.className = 'message-body';

    if (role === 'assistant') {
      renderFormattedContent(body, text);
    } else {
      body.textContent = text;
    }

    if (citations && citations.length > 0) {
      const citationsWrapper = document.createElement('div');
      citationsWrapper.className = 'citations-wrapper';
      citationsWrapper.innerHTML = buildExcerptsDrawerHtml(citations);
      body.appendChild(citationsWrapper);
    }

    if (modelBadge && role === 'assistant') {
      const badgeDiv = document.createElement('div');
      badgeDiv.style.cssText = 'font-size: 11px; color: #64748b; margin-top: 6px; font-family: monospace; display: flex; align-items: center; gap: 4px; border-top: 1px solid #334155; padding-top: 4px;';
      badgeDiv.innerHTML = `⚡ Engine Model: <strong style="color:#3b82f6;">${modelBadge}</strong>`;
      body.appendChild(badgeDiv);
    }

    // Attach citation pill and copy handlers
    body.addEventListener('click', (e) => {
      const pill = e.target.closest('.citation-pill');
      if (pill) {
        const idx = pill.getAttribute('data-source-index');
        const drawer = body.querySelector('.sources-drawer');
        if (drawer) drawer.open = true;
        const target = body.querySelector(`#source-card-${idx}`);
        if (target) {
          target.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          target.classList.add('highlighted');
          setTimeout(() => target.classList.remove('highlighted'), 2500);
        }
      }
      const copyBtn = e.target.closest('.btn-copy-excerpt');
      if (copyBtn) {
        const card = copyBtn.closest('.excerpt-card');
        const txt = card ? card.querySelector('.excerpt-text')?.innerText : '';
        if (txt) {
          navigator.clipboard.writeText(txt);
          copyBtn.textContent = '✔️ Copied!';
          setTimeout(() => { copyBtn.textContent = '📋 Copy'; }, 2000);
        }
      }
    });

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(body);
    if (elements.chatMessages) {
      elements.chatMessages.appendChild(msgDiv);
      elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }
  }

  function createStreamingMessage(citations = [], initialModel = '') {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message system-message streaming-active';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '⚡';

    const body = document.createElement('div');
    body.className = 'message-body';

    const textContainer = document.createElement('div');
    textContainer.className = 'streaming-text-container';
    body.appendChild(textContainer);

    const cursor = document.createElement('span');
    cursor.className = 'streaming-cursor';
    body.appendChild(cursor);

    let citationsWrapper = null;
    if (citations && citations.length > 0) {
      citationsWrapper = document.createElement('div');
      citationsWrapper.className = 'citations-wrapper';
      citationsWrapper.innerHTML = buildExcerptsDrawerHtml(citations);
      body.appendChild(citationsWrapper);
    }

    const badgeDiv = document.createElement('div');
    badgeDiv.style.cssText = 'font-size: 11px; color: #64748b; margin-top: 6px; font-family: monospace; display: flex; align-items: center; gap: 4px; border-top: 1px solid #334155; padding-top: 4px;';
    badgeDiv.innerHTML = `⚡ Engine Model: <strong style="color:#3b82f6;">${initialModel || state.model}</strong>`;
    body.appendChild(badgeDiv);

    body.addEventListener('click', (e) => {
      const pill = e.target.closest('.citation-pill');
      if (pill) {
        const idx = pill.getAttribute('data-source-index');
        const drawer = body.querySelector('.sources-drawer');
        if (drawer) drawer.open = true;
        const target = body.querySelector(`#source-card-${idx}`);
        if (target) {
          target.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          target.classList.add('highlighted');
          setTimeout(() => target.classList.remove('highlighted'), 2500);
        }
      }
      const copyBtn = e.target.closest('.btn-copy-excerpt');
      if (copyBtn) {
        const card = copyBtn.closest('.excerpt-card');
        const txt = card ? card.querySelector('.excerpt-text')?.innerText : '';
        if (txt) {
          navigator.clipboard.writeText(txt);
          copyBtn.textContent = '✔️ Copied!';
          setTimeout(() => { copyBtn.textContent = '📋 Copy'; }, 2000);
        }
      }
    });

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(body);
    if (elements.chatMessages) {
      elements.chatMessages.appendChild(msgDiv);
      elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }

    let accumulated = '';
    let lastRenderTime = 0;

    return {
      update: (chunk, fullText) => {
        accumulated = fullText;
        const now = Date.now();
        // Throttle full markdown/KaTeX parsing during rapid streaming tokens for smooth 60fps UI
        if (now - lastRenderTime > 80) {
          renderFormattedContent(textContainer, accumulated);
          lastRenderTime = now;
          if (elements.chatMessages) elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
        }
      },
      finish: (modelBadge) => {
        cursor.remove();
        msgDiv.classList.remove('streaming-active');
        renderFormattedContent(textContainer, accumulated);
        if (modelBadge) {
          badgeDiv.innerHTML = `⚡ Engine Model: <strong style="color:#3b82f6;">${modelBadge}</strong>`;
        }
        if (elements.chatMessages) elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
      },
      remove: () => {
        msgDiv.remove();
      }
    };
  }

  function showThinkingIndicator(initialStatus = 'Searching engineering library...') {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message system-message thinking-message';
    msgDiv.id = 'active-thinking-message';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '⚡';

    const body = document.createElement('div');
    body.className = 'message-body';
    body.style.cssText = 'display:flex; align-items:center; gap:8px; color:var(--text-muted, #94a3b8); font-style:italic; font-size:14px;';
    body.innerHTML = `
      <span class="spinner" style="display:inline-block; width:14px; height:14px; border:2px solid #3b82f6; border-top-color:transparent; border-radius:50%; animation:spin 0.8s linear infinite;"></span>
      <span id="thinking-status-text">${initialStatus}</span>
    `;

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(body);
    if (elements.chatMessages) {
      elements.chatMessages.appendChild(msgDiv);
      elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }

    return {
      update: (text) => {
        const el = document.getElementById('thinking-status-text');
        if (el) el.textContent = text;
        if (elements.chatMessages) elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
      },
      remove: () => {
        msgDiv.remove();
      }
    };
  }


  // --- Event Listeners ---

  function setupEventListeners() {
    if (elements.btnSettings) {
      elements.btnSettings.addEventListener('click', () => {
        elements.apiKeyInput.value = state.apiKey;
        if (elements.chatModelSelect) elements.chatModelSelect.value = state.model || 'gemini-2.0-flash';
        if (elements.cloudUrlInput) elements.cloudUrlInput.value = state.cloudUrl;
        elements.settingsModal.classList.remove('hidden');
      });
    }

    const syncModelChange = (val) => {
      if (!val) return;
      state.model = val;
      localStorage.setItem('gemini_model', val);
      if (elements.chatModelSelect) elements.chatModelSelect.value = val;
      if (elements.modelSelect) elements.modelSelect.value = val;
    };

    if (elements.chatModelSelect) {
      elements.chatModelSelect.addEventListener('change', (e) => syncModelChange(e.target.value));
    }
    if (elements.modelSelect) {
      elements.modelSelect.addEventListener('change', (e) => syncModelChange(e.target.value));
    }




    if (elements.settingsModal) {
      const closeBtn = elements.settingsModal.querySelector('.modal-close');
      if (closeBtn) {
        closeBtn.addEventListener('click', () => {
          elements.settingsModal.classList.add('hidden');
        });
      }
    }

    if (elements.btnFetchCloud) {
      elements.btnFetchCloud.addEventListener('click', async () => {
        const cloudUrl = elements.cloudUrlInput ? elements.cloudUrlInput.value.trim() : '';
        state.apiKey = elements.apiKeyInput.value.trim();
        state.cloudUrl = cloudUrl;
        localStorage.setItem('gemini_api_key', state.apiKey);
        localStorage.setItem('cloud_storage_url', cloudUrl);

        if (!cloudUrl) {
          elements.settingsStatus.className = 'status-msg error';
          elements.settingsStatus.textContent = 'Please enter a cloud share link or PDF URL.';
          elements.settingsStatus.classList.remove('hidden');
          return;
        }

        await scanGoogleDriveFolder(cloudUrl);
      });
    }

    if (elements.btnSelectAllDrive) {
      elements.btnSelectAllDrive.addEventListener('click', () => {
        document.querySelectorAll('.drive-file-cb').forEach(cb => cb.checked = true);
      });
    }

    if (elements.btnDeselectAllDrive) {
      elements.btnDeselectAllDrive.addEventListener('click', () => {
        document.querySelectorAll('.drive-file-cb').forEach(cb => cb.checked = false);
      });
    }

    if (elements.btnImportSelectedDrive) {
      elements.btnImportSelectedDrive.addEventListener('click', async () => {
        await importSelectedDriveFiles();
      });
    }

    if (elements.btnSaveKey) {
      elements.btnSaveKey.addEventListener('click', async () => {
        state.apiKey = elements.apiKeyInput.value.trim();
        if (elements.chatModelSelect) state.model = elements.chatModelSelect.value;
        state.cloudUrl = elements.cloudUrlInput ? elements.cloudUrlInput.value.trim() : '';
        localStorage.setItem('gemini_api_key', state.apiKey);
        localStorage.setItem('gemini_model', state.model);
        localStorage.setItem('cloud_storage_url', state.cloudUrl);
        await fetchRepoKB();
        await discoverUserModels(state.apiKey);
        elements.settingsStatus.className = 'status-msg success';
        elements.settingsStatus.textContent = 'Settings saved! Database updated.';
        elements.settingsStatus.classList.remove('hidden');
        setTimeout(() => {
          elements.settingsModal.classList.add('hidden');
          elements.settingsStatus.classList.add('hidden');
        }, 1200);
      });
    }


    if (elements.btnTestKey) {
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

        const discovered = await discoverUserModels(testKey);
        let apiErr = '';

        let chatStatus = false;
        let embedStatus = false;

        const chatModels = [
          ...discovered,
          elements.chatModelSelect?.value,
          elements.modelSelect?.value,
          state.model
        ].filter((m, i, arr) => m && arr.indexOf(m) === i);


        let workingChatModel = '';

        for (const cm of chatModels) {
          try {
            const url = `https://generativelanguage.googleapis.com/v1beta/models/${cm}:generateContent`;
            let res = await fetch(url, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json', 'x-goog-api-key': testKey },
              body: JSON.stringify({ contents: [{ role: 'user', parts: [{ text: 'Ping' }] }] })
            });

            if (!res.ok) {
              res = await fetch(`${url}?key=${testKey}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ contents: [{ role: 'user', parts: [{ text: 'Ping' }] }] })
              });
            }

            if (res.ok) {
              chatStatus = true;
              workingChatModel = cm;
              break;
            } else {
              const errData = await res.json().catch(() => ({}));
              apiErr = errData.error?.message || `HTTP ${res.status}`;
            }
          } catch (e) {
            apiErr = e.message;
          }
        }

        const embedModels = ['gemini-embedding-2', 'text-embedding-004'];
        let workingEmbedModel = '';

        for (const em of embedModels) {
          try {
            const url = `https://generativelanguage.googleapis.com/v1beta/models/${em}:embedContent`;
            let res = await fetch(url, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json', 'x-goog-api-key': testKey },
              body: JSON.stringify({ model: `models/${em}`, content: { parts: [{ text: 'Ping' }] } })
            });

            if (!res.ok) {
              res = await fetch(`${url}?key=${testKey}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: `models/${em}`, content: { parts: [{ text: 'Ping' }] } })
              });
            }

            if (res.ok) {
              embedStatus = true;
              workingEmbedModel = em;
              break;
            }
          } catch (e) {}
        }

        if (chatStatus) {
          if (workingChatModel) {
            state.model = workingChatModel;
            localStorage.setItem('gemini_model', workingChatModel);
            if (elements.chatModelSelect) elements.chatModelSelect.value = workingChatModel;
          }
          elements.settingsStatus.className = 'status-msg success';
          elements.settingsStatus.textContent = `✔️ Chat Ready (${workingChatModel}) | ${embedStatus ? `✔️ Embedder Ready (${workingEmbedModel})` : '⚠️ Keyword-Only Retrieval'}`;
        } else {

          elements.settingsStatus.className = 'status-msg error';
          elements.settingsStatus.textContent = `❌ API Error: ${apiErr || 'Invalid API Key or network blocked'}. Please check your key at aistudio.google.com.`;
        }
      });
    }


    if (elements.btnLibrary) {
      elements.btnLibrary.addEventListener('click', () => {
        updateLibraryUI();
        elements.libraryModal.classList.remove('hidden');
      });
    }

    if (elements.libraryModal) {
      const closeBtn = elements.libraryModal.querySelector('.modal-close');
      if (closeBtn) {
        closeBtn.addEventListener('click', () => {
          elements.libraryModal.classList.add('hidden');
        });
      }
    }

    if (elements.btnUpload && elements.fileInput) {
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
        populateDocScopeSelect();
        elements.fileInput.value = '';
      });
    }

    if (elements.filterPills) {
      elements.filterPills.forEach((pill) => {
        pill.addEventListener('click', (e) => {
          elements.filterPills.forEach((p) => p.classList.remove('active'));
          e.currentTarget.classList.add('active');
          state.activeFilter = e.currentTarget.getAttribute('data-filter');
          if (elements.searchInput && elements.searchInput.value.trim()) {
            elements.searchInput.dispatchEvent(new Event('input'));
          }
        });
      });
    }

    if (elements.searchInput) {
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
          const docScope = elements.docScopeSelect ? elements.docScopeSelect.value : 'all';
          const results = await performHybridSearch(val, 6, docScope);
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
    }

    if (elements.searchClearBtn) {
      elements.searchClearBtn.addEventListener('click', () => {
        elements.searchInput.value = '';
        elements.searchClearBtn.classList.add('hidden');
        elements.searchResultsSection.classList.add('hidden');
      });
    }

    if (elements.closeResultsBtn) {
      elements.closeResultsBtn.addEventListener('click', () => {
        elements.searchResultsSection.classList.add('hidden');
      });
    }

    if (elements.btnClearChat) {
      elements.btnClearChat.addEventListener('click', () => {
        state.chatHistory = [];
        if (elements.chatMessages) {
          elements.chatMessages.innerHTML = '';
          renderMessage('assistant', '⚡ **New Chat Session Started.** Ask any engineering question or select a specific code from the dropdown below.');
        }
      });
    }

    if (elements.chatInput) {
      elements.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          if (elements.chatForm) {
            elements.chatForm.requestSubmit ? elements.chatForm.requestSubmit() : elements.chatForm.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
          }
        }
      });
    }

    if (elements.chatForm) {
      elements.chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const prompt = elements.chatInput.value.trim();
        if (!prompt) return;

        elements.chatInput.value = '';
        renderMessage('user', prompt);

        const useRag = elements.useRagToggle.checked;
        const docScope = elements.docScopeSelect ? elements.docScopeSelect.value : 'all';
        let retrieved = [];

        const thinking = showThinkingIndicator(useRag ? '🔍 Searching engineering library & matching clauses...' : '⚡ Connecting to Gemini AI...');

        let streamer = null;
        try {
          if (useRag) {
            retrieved = await performHybridSearch(prompt, 6, docScope);
            thinking.update(`🧠 Synthesizing with ${state.model || 'Gemini'}... (${retrieved.length} clauses retrieved)`);
          } else {
            thinking.update(`🧠 Synthesizing with ${state.model || 'Gemini'}...`);
          }

          thinking.remove();
          streamer = createStreamingMessage(retrieved, state.model);

          const res = await streamGeminiChat(prompt, retrieved, docScope, (delta, full) => {
            if (streamer) streamer.update(delta, full);
          });

          streamer.finish(res.modelUsed);

          // Retain conversational memory
          state.chatHistory.push({ role: 'user', content: prompt });
          state.chatHistory.push({ role: 'assistant', content: res.answer });
          if (state.chatHistory.length > 16) {
            state.chatHistory = state.chatHistory.slice(-16);
          }
        } catch (err) {
          thinking.remove();
          if (streamer) streamer.remove();
          renderMessage('assistant', `⚠️ **Error**: ${err.message}`);
        }
      });
    }


    if (elements.btnExportDb) {
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
    }

    if (elements.btnImportDb && elements.importDbInput) {
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
            for (const doc of data.documents) {
              const docChunks = data.chunks.filter((c) => c.docId === doc.id || c.file === doc.filename);
              await saveLocalDocument(doc, docChunks);
            }
            alert('Backup restored successfully!');
            updateLibraryUI();
            populateDocScopeSelect();
          }
        } catch (err) {
          alert('Invalid backup file: ' + err.message);
        }
      });
    }
  }

  async function init() {
    initElements();
    setupEventListeners();
    initIndexedDB();
    fetchRepoKB();
    if (state.apiKey) {
      discoverUserModels(state.apiKey);
    }
    console.log('Researcher AI Web App initialized with Selective Checklist UI.');
  }

  window.addEventListener('DOMContentLoaded', init);
})();
