// Researcher — Client-Side Engineering RAG Web App
// 100% pure client-side for GitHub Pages, matching the local desktop app experience.

const $ = id => document.getElementById(id);
const esc = s => (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// Configure marked with breaks and KaTeX math extension
if (window.marked) {
  marked.setOptions({ breaks: true });
  if (window.markedKatex) {
    marked.use(window.markedKatex({ throwOnError: false, nonStandard: true }));
  }
}

// Global State
let db = null;
let documents = [];        // [{id, filename, format, pages, chunk_count, status, is_repo}]
let chunks = [];           // [{id, doc_id, filename, text, page, clause, tokens}]
let docSel = {};           // doc_id -> boolean (included in chat scope)
let chats = [];            // [{id, title, messages: [{role, content, sources, tps}], created_at}]
let activeChatId = null;
let isBusy = false;
let abortController = null;

const DEFAULT_SETTINGS = {
  apiKey: '',
  model: 'gemini-3-flash-preview',
  topK: 6,
  vectorWeight: 0.5
};

let settings = { ...DEFAULT_SETTINGS };

// Load settings from localStorage
function loadSettings() {
  try {
    const saved = localStorage.getItem('researcher_settings');
    if (saved) {
      settings = { ...DEFAULT_SETTINGS, ...JSON.parse(saved) };
    }
  } catch (e) {
    console.error('Failed to load settings:', e);
  }
}

function saveSettings() {
  try {
    localStorage.setItem('researcher_settings', JSON.stringify(settings));
  } catch (e) {
    console.error('Failed to save settings:', e);
  }
}

// ----------------------------------------------------
// IndexedDB Storage
// ----------------------------------------------------
function openDatabase() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('researcher_online_v1', 1);
    req.onupgradeneeded = e => {
      const d = e.target.result;
      if (!d.objectStoreNames.contains('documents')) {
        d.createObjectStore('documents', { keyPath: 'id' });
      }
      if (!d.objectStoreNames.contains('chunks')) {
        const chunkStore = d.createObjectStore('chunks', { keyPath: 'id' });
        chunkStore.createIndex('doc_id', 'doc_id', { unique: false });
      }
    };
    req.onsuccess = e => {
      db = e.target.result;
      resolve(db);
    };
    req.onerror = e => reject(e.target.error);
  });
}

function dbSaveDocAndChunks(doc, docChunks) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(['documents', 'chunks'], 'readwrite');
    const dStore = tx.objectStore('documents');
    const cStore = tx.objectStore('chunks');
    dStore.put(doc);
    for (const c of docChunks) {
      cStore.put(c);
    }
    tx.oncomplete = () => resolve();
    tx.onerror = e => reject(e.target.error);
  });
}

function dbDeleteDoc(docId) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(['documents', 'chunks'], 'readwrite');
    const dStore = tx.objectStore('documents');
    const cStore = tx.objectStore('chunks');
    dStore.delete(docId);
    const idx = cStore.index('doc_id');
    const range = IDBKeyRange.only(docId);
    const req = idx.openCursor(range);
    req.onsuccess = e => {
      const cursor = e.target.result;
      if (cursor) {
        cursor.delete();
        cursor.continue();
      }
    };
    tx.oncomplete = () => resolve();
    tx.onerror = e => reject(e.target.error);
  });
}

function dbClearLocalDocs() {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(['documents', 'chunks'], 'readwrite');
    tx.objectStore('documents').clear();
    tx.objectStore('chunks').clear();
    tx.oncomplete = () => resolve();
    tx.onerror = e => reject(e.target.error);
  });
}

function dbLoadAll() {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(['documents', 'chunks'], 'readonly');
    const dStore = tx.objectStore('documents');
    const cStore = tx.objectStore('chunks');
    const dReq = dStore.getAll();
    const cReq = cStore.getAll();
    let loadedDocs = [], loadedChunks = [];
    dReq.onsuccess = () => { loadedDocs = dReq.result || []; };
    cReq.onsuccess = () => { loadedChunks = cReq.result || []; };
    tx.oncomplete = () => resolve({ docs: loadedDocs, chunks: loadedChunks });
    tx.onerror = e => reject(e.target.error);
  });
}

// ----------------------------------------------------
// Load Initial Documents (Repository + Local IndexedDB)
// ----------------------------------------------------
async function initializeLibrary() {
  try {
    await openDatabase();
    const localData = await dbLoadAll();

    // Check if repo knowledge_base.json exists
    let repoDocs = [];
    let repoChunks = [];
    try {
      const resp = await fetch('knowledge_base.json', { cache: 'no-cache' });
      if (resp.ok) {
        const kb = await resp.json();
        if (kb.documents && Array.isArray(kb.documents)) {
          repoDocs = kb.documents.map(d => ({ ...d, is_repo: true }));
        }
        if (kb.chunks && Array.isArray(kb.chunks)) {
          repoChunks = kb.chunks.map(c => ({
            ...c,
            tokens: tokenize(c.text || '')
          }));
        }
      }
    } catch (e) {
      console.log('No repository knowledge_base.json found or failed to load:', e);
    }

    // Merge repo and local documents
    documents = [...repoDocs, ...localData.docs];
    chunks = [...repoChunks, ...localData.chunks];

    // Initialize doc selection (default to checked)
    for (const d of documents) {
      if (!(d.id in docSel)) {
        docSel[d.id] = true;
      }
    }

    refreshDocsList();
    updateStats();
  } catch (err) {
    console.error('Library initialization error:', err);
    $('embDot').className = 'dot bad';
    $('embStatus').textContent = 'library error';
  }
}

// ----------------------------------------------------
// Document Selection & Scoping
// ----------------------------------------------------
function selectedDocIds() {
  return Object.keys(docSel).filter(id => docSel[id]);
}

function updateSelInfo() {
  const total = documents.length;
  const sel = selectedDocIds().length;
  $('selInfo').textContent = total ? `chat scope: ${sel === total ? 'all' : sel + '/' + total} docs ·` : '';
}

function toggleSel(id, val) {
  docSel[id] = val;
  const row = document.querySelector(`.doc[data-id="${id}"]`);
  if (row) {
    row.classList.toggle('excluded', !val);
  }
  updateSelInfo();
}

function selectAll(val) {
  for (const d of documents) {
    docSel[d.id] = val;
  }
  document.querySelectorAll('.doc').forEach(el => el.classList.toggle('excluded', !val));
  document.querySelectorAll('.doc input[type=checkbox]').forEach(cb => cb.checked = val);
  updateSelInfo();
}

async function clearLocalDocs() {
  const localDocs = documents.filter(d => !d.is_repo);
  if (localDocs.length === 0) {
    alert('No local documents to clear.');
    return;
  }
  if (!confirm(`Delete all ${localDocs.length} uploaded documents from this device?`)) return;

  await dbClearLocalDocs();
  documents = documents.filter(d => d.is_repo);
  chunks = chunks.filter(c => documents.some(d => d.id === c.doc_id));
  docSel = {};
  for (const d of documents) docSel[d.id] = true;
  refreshDocsList();
  updateStats();
}

async function deleteDocument(id) {
  const doc = documents.find(d => String(d.id) === String(id));
  if (!doc) return;
  if (doc.is_repo) {
    alert('Pre-indexed repository standards cannot be deleted from the browser. You can uncheck them to exclude them from chat.');
    return;
  }
  if (!confirm(`Remove "${doc.filename}" from library?`)) return;

  await dbDeleteDoc(doc.id);
  documents = documents.filter(d => d.id !== doc.id);
  chunks = chunks.filter(c => c.doc_id !== doc.id);
  delete docSel[doc.id];
  refreshDocsList();
  updateStats();
}

function refreshDocsList() {
  const list = $('docList');
  list.innerHTML = '';

  for (const d of documents) {
    const isChecked = docSel[d.id] !== false;
    const div = document.createElement('div');
    div.className = 'doc' + (isChecked ? '' : ' excluded');
    div.dataset.id = d.id;

    const pageLabel = d.pages ? `${d.pages} pages • ` : '';
    const meta = `${(d.format || 'doc').toUpperCase()} • ${pageLabel}${d.chunk_count || 0} chunks${d.is_repo ? ' • 📦 repo' : ''}`;
    const badge = d.status === 'ready' ? 'ready' : (d.status || 'ready');
    const cls = badge === 'ready' ? 'b-ready' : 'b-kw';

    div.innerHTML = `
      <input type="checkbox" title="Include in chat" ${isChecked ? 'checked' : ''} onchange="toggleSel('${d.id}', this.checked)">
      <div style="flex:1;min-width:0">
        <div class="name" title="${esc(d.filename)}">${esc(d.filename)}</div>
        <div class="meta">${meta}</div>
      </div>
      <span class="badge ${cls}">${badge}</span>
      ${!d.is_repo ? `<button title="Delete" onclick="deleteDocument('${d.id}')">✕</button>` : ''}
    `;
    list.appendChild(div);
  }

  updateSelInfo();
}

function updateStats() {
  const docCount = documents.length;
  const chunkCount = chunks.length;
  $('stats').textContent = `${docCount} documents · ${chunkCount} chunks · hybrid ready`;
  $('embDot').className = 'dot ok';
  $('embStatus').textContent = `library ready (${docCount} docs)`;
}

// ----------------------------------------------------
// File Upload & Client-Side Extraction
// ----------------------------------------------------
const uploadZone = $('uploadZone');
const fileInput = $('fileInput');

uploadZone.onclick = () => fileInput.click();
fileInput.onchange = () => handleFiles(fileInput.files);

uploadZone.ondragover = e => {
  e.preventDefault();
  uploadZone.classList.add('drag');
};
uploadZone.ondragleave = () => uploadZone.classList.remove('drag');
uploadZone.ondrop = e => {
  e.preventDefault();
  uploadZone.classList.remove('drag');
  handleFiles(e.dataTransfer.files);
};

async function handleFiles(fileList) {
  if (!fileList || fileList.length === 0) return;
  uploadZone.textContent = 'Parsing & indexing documents…';

  try {
    for (const file of fileList) {
      await processUploadedFile(file);
    }
  } catch (err) {
    console.error('Error processing file:', err);
    alert('Failed to process document: ' + err.message);
  } finally {
    uploadZone.innerHTML = '⬆ Drop documents here or click to browse<br><span style="font-size:11px">PDF · Word · Excel · text</span>';
    fileInput.value = '';
    refreshDocsList();
    updateStats();
  }
}

async function processUploadedFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  let text = '';
  let pages = 1;

  if (ext === 'pdf') {
    const arrayBuffer = await file.arrayBuffer();
    if (!window.pdfjsLib) {
      throw new Error('PDF.js library is not loaded');
    }
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    pages = pdf.numPages;
    const pageTexts = [];
    for (let i = 1; i <= pages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      const strings = content.items.map(item => item.str);
      pageTexts.push(`[Page ${i}]\n` + strings.join(' '));
    }
    text = pageTexts.join('\n\n');
  } else if (ext === 'docx') {
    const arrayBuffer = await file.arrayBuffer();
    if (!window.mammoth) {
      throw new Error('Mammoth.js library is not loaded');
    }
    const result = await mammoth.extractRawText({ arrayBuffer });
    text = result.value;
  } else {
    // Text, Markdown, CSV, JSON
    text = await file.text();
  }

  if (!text || text.trim().length === 0) {
    throw new Error(`No extractable text found in "${file.name}"`);
  }

  const docId = 'doc_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6);
  const docChunks = splitIntoChunks(docId, file.name, text, pages);

  const docRecord = {
    id: docId,
    filename: file.name,
    format: ext,
    pages: pages,
    chunk_count: docChunks.length,
    status: 'ready',
    is_repo: false,
    uploaded_at: new Date().toISOString()
  };

  await dbSaveDocAndChunks(docRecord, docChunks);
  documents.push(docRecord);
  chunks.push(...docChunks);
  docSel[docId] = true;
}

// Semantic Chunking with Clause Detection
function splitIntoChunks(docId, filename, fullText, totalPages) {
  const result = [];
  const lines = fullText.split(/\r?\n/);
  let curChunk = '';
  let curPage = 1;
  let curClause = '';
  let chunkIdx = 0;

  const flush = () => {
    if (curChunk.trim().length > 30) {
      result.push({
        id: `${docId}_c${chunkIdx++}`,
        doc_id: docId,
        filename: filename,
        text: curChunk.trim(),
        page: curPage,
        clause: curClause || `Section ${chunkIdx}`,
        tokens: tokenize(curChunk)
      });
      // 15% overlap for continuity
      const words = curChunk.trim().split(/\s+/);
      curChunk = words.slice(Math.max(0, words.length - 25)).join(' ') + '\n';
    }
  };

  for (const line of lines) {
    // Detect page markers
    const pgMatch = line.match(/\[Page\s+(\d+)\]/i);
    if (pgMatch) {
      curPage = parseInt(pgMatch[1], 10);
      continue;
    }

    // Detect clause/section headings (e.g. "Article 3.1", "304.1.2", "Section 4", "Table 2")
    const clauseMatch = line.match(/^([A-Z0-9.\-_]{2,15}(\s+[A-Za-z0-9.\-_]+){0,4})/);
    if (clauseMatch && line.length < 80 && line.trim().length > 2) {
      curClause = clauseMatch[1].trim();
    }

    curChunk += line + '\n';
    if (curChunk.length >= 850) {
      flush();
    }
  }
  flush();

  return result;
}

function tokenize(text) {
  return (text.toLowerCase().match(/[a-z0-9_\-\.]{2,}/g) || []);
}

// ----------------------------------------------------
// Scoped Search & RAG Retrieval
// ----------------------------------------------------
function retrieveContext(query, topK = 6) {
  const allowedDocIds = new Set(selectedDocIds().map(String));
  if (allowedDocIds.size === 0) {
    return [];
  }

  // Filter chunks by selected docs
  const scopedChunks = chunks.filter(c => allowedDocIds.has(String(c.doc_id)));
  if (scopedChunks.length === 0) return [];

  const qTokens = tokenize(query);
  if (qTokens.length === 0) {
    return scopedChunks.slice(0, topK);
  }

  // Clause boost terms (numbers, decimals, code designations)
  const isClauseTerm = t => /\d/.test(t) || t.length > 5;

  const scored = [];
  for (let i = 0; i < scopedChunks.length; i++) {
    const c = scopedChunks[i];
    const chunkTokens = c.tokens || tokenize(c.text);
    const tokenSet = new Set(chunkTokens);

    let score = 0;
    for (const q of qTokens) {
      if (tokenSet.has(q)) {
        score += isClauseTerm(q) ? 3.0 : 1.0;
      } else if (c.text.toLowerCase().includes(q)) {
        score += 0.5;
      }
    }

    // Boost if query matches clause header
    if (c.clause) {
      const clLower = c.clause.toLowerCase();
      for (const q of qTokens) {
        if (clLower.includes(q)) score += 4.0;
      }
    }

    if (score > 0) {
      scored.push({ chunk: c, score, index: i });
    }
  }

  scored.sort((a, b) => b.score - a.score);

  // Fallback: If broad question has very few matches, include introduction chunks
  if (scored.length < 2) {
    const fallbackMap = new Map();
    for (const docId of allowedDocIds) {
      const docLeadChunks = scopedChunks.filter(c => String(c.doc_id) === String(docId)).slice(0, 3);
      for (const fc of docLeadChunks) {
        fallbackMap.set(fc.id, fc);
      }
    }
    for (const s of scored) {
      fallbackMap.set(s.chunk.id, s.chunk);
    }
    return Array.from(fallbackMap.values()).slice(0, topK);
  }

  // Neighbor expansion: Add immediate adjacent chunks for top 3 matches to prevent cutting clauses
  const selectedMap = new Map();
  for (const item of scored.slice(0, topK)) {
    selectedMap.set(item.chunk.id, item.chunk);

    // Expand previous chunk
    if (item.index > 0) {
      const prev = scopedChunks[item.index - 1];
      if (prev && prev.doc_id === item.chunk.doc_id && !selectedMap.has(prev.id)) {
        selectedMap.set(prev.id, prev);
      }
    }
    // Expand next chunk
    if (item.index < scopedChunks.length - 1) {
      const next = scopedChunks[item.index + 1];
      if (next && next.doc_id === item.chunk.doc_id && !selectedMap.has(next.id)) {
        selectedMap.set(next.id, next);
      }
    }
  }

  return Array.from(selectedMap.values()).slice(0, topK + 3);
}

// ----------------------------------------------------
// Chat & Gemini Streaming Engine
// ----------------------------------------------------
const SYSTEM_PROMPT = `You are a Senior Engineering Codes & Standards Specialist.
Your job is to provide accurate, definitive, and strictly grounded engineering analysis based on the retrieved code excerpts.

CRITICAL RULES:
1. Always cite exact clauses, paragraphs, sections, and page numbers where available. Use bracket citations like [1], [2] corresponding to the excerpts.
2. If equations or formulas apply, write them cleanly using LaTeX format (e.g. $t = \\frac{PD}{2(SEW + PY)}$).
3. If specific design factors, material allowable stresses, or temperature limits are requested, extract and tabulate them clearly.
4. When comparing multiple codes, present a clear markdown table highlighting differences, scope applicability, and testing criteria.
5. If the excerpts do not contain the answer, explicitly state what is missing rather than inventing values.`;

async function streamChatResponse(userPrompt) {
  if (!settings.apiKey) {
    openSettingsModal();
    alert('Please enter your Google Gemini API Key in Settings to start chatting.');
    return;
  }

  const useRAG = $('useLibrary').checked;
  const isCompare = $('compareMode').checked;

  let contextExcerpts = [];
  if (useRAG) {
    contextExcerpts = retrieveContext(userPrompt, settings.topK || 6);
  }

  // Add User Message to Chat UI
  addMessageToUI('user', userPrompt);
  $('input').value = '';
  $('input').style.height = 'auto';

  // Build Assistant Message UI Container
  const assistantBody = addMessageToUI('assistant', '<span class="typing">Thinking…</span>');

  // Build Context Header
  let augmentedPrompt = userPrompt;
  if (contextExcerpts.length > 0) {
    let contextStr = '--- RETRIEVED CODES & STANDARDS EXCERPTS ---\n\n';
    contextExcerpts.forEach((c, idx) => {
      contextStr += `[${idx + 1}] Document: "${c.filename}" | Clause: ${c.clause || 'General'} | Page: ${c.page || 'N/A'}\n${c.text}\n\n`;
    });
    contextStr += '--- END EXCERPTS ---\n\n';

    if (isCompare) {
      contextStr += 'Instruction: Compare the selected documents in detail regarding the user question.\n\n';
    }
    contextStr += `User Question: ${userPrompt}\nAnswer strictly based on the excerpts with [1], [2] citations:`;
    augmentedPrompt = contextStr;
  }

  // Prepare Gemini Request with Candidate Fallbacks & Rate-Limit Retry
  const requestedModel = settings.model || 'gemini-3-flash-preview';
  const candidateModels = [requestedModel];
  for (const m of ['gemini-3-flash-preview', 'gemini-3.7-flash', 'gemini-2.0-flash']) {
    if (!candidateModels.includes(m)) candidateModels.push(m);
  }

  // History payload
  const contents = [];
  const currentChat = getActiveChat();
  if (currentChat && currentChat.messages.length > 0) {
    // Include last 4 turns for multi-turn context
    const recent = currentChat.messages.slice(-6);
    for (const m of recent) {
      contents.push({
        role: m.role === 'user' ? 'user' : 'model',
        parts: [{ text: m.content }]
      });
    }
  }
  contents.push({
    role: 'user',
    parts: [{ text: augmentedPrompt }]
  });

  const payload = {
    contents: contents,
    systemInstruction: {
      parts: [{ text: SYSTEM_PROMPT }]
    },
    generationConfig: {
      temperature: 0.2,
      maxOutputTokens: 3000
    }
  };

  isBusy = true;
  $('sendBtn').disabled = true;
  $('sendBtn').textContent = '⏹';
  $('tpsWrap').textContent = 'connecting…';
  const startTime = performance.now();
  let totalTokens = 0;
  let fullResponseText = '';
  let activeModelUsed = requestedModel;

  abortController = new AbortController();

  try {
    let resp = null;
    let lastErrorMsg = '';

    for (const modelName of candidateModels) {
      activeModelUsed = modelName;
      const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:streamGenerateContent?alt=sse&key=${settings.apiKey}`;

      let attempt = 0;
      while (attempt < 2) {
        attempt++;
        try {
          resp = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: abortController.signal
          });

          if (resp.ok) {
            break;
          }

          const errJson = await resp.json().catch(() => ({}));
          const msg = errJson.error?.message || `HTTP ${resp.status} ${resp.statusText}`;
          lastErrorMsg = msg;

          // Check if rate limited and told to wait
          if (resp.status === 429 || msg.toLowerCase().includes('quota') || msg.toLowerCase().includes('rate')) {
            const retryMatch = msg.match(/retry in\s*(\d+(?:\.\d+)?)\s*ms/i);
            const waitMs = retryMatch ? parseFloat(retryMatch[1]) : 0;
            if (attempt === 1 && waitMs > 0 && waitMs <= 2500) {
              $('tpsWrap').textContent = `retrying in ${(waitMs/1000).toFixed(1)}s…`;
              await new Promise(r => setTimeout(r, waitMs + 100));
              continue;
            }
          }
          break; // move to next candidate model
        } catch (fetchErr) {
          if (fetchErr.name === 'AbortError') throw fetchErr;
          lastErrorMsg = fetchErr.message;
          break;
        }
      }

      if (resp && resp.ok) {
        break; // Successfully connected to a model
      }
    }

    if (!resp || !resp.ok) {
      throw new Error(lastErrorMsg || 'Failed to connect to Gemini after trying fallback models.');
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // Keep unfinished line

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.replace('data: ', '').trim();
          if (!jsonStr) continue;

          try {
            const data = JSON.parse(jsonStr);
            const candidates = data.candidates || [];
            if (candidates[0]?.content?.parts) {
              for (const part of candidates[0].content.parts) {
                // Filter thought tokens
                if (part.thought) continue;
                if (part.text) {
                  fullResponseText += part.text;
                  totalTokens += Math.ceil(part.text.length / 4);

                  const elapsed = (performance.now() - startTime) / 1000;
                  const tps = elapsed > 0 ? (totalTokens / elapsed).toFixed(1) : '0';
                  $('tpsWrap').textContent = `${tps} tok/s · ${totalTokens} tok`;

                  // Live render markdown
                  assistantBody.innerHTML = renderMarkdown(fullResponseText);
                  renderMathInElement(assistantBody);
                }
              }
            }
          } catch (pe) {
            console.error('SSE parse error:', pe);
          }
        }
      }
    }

    // Finalize response
    if (activeModelUsed !== requestedModel && fullResponseText) {
      fullResponseText = `> [!NOTE]\n> *Free tier quota reached for \`${requestedModel}\`. Automatically answered using \`${activeModelUsed}\`.*\n\n` + fullResponseText;
    }

    if (!fullResponseText) {
      fullResponseText = 'No text returned by the model.';
      assistantBody.innerHTML = fullResponseText;
    } else {
      assistantBody.innerHTML = renderMarkdown(fullResponseText);
      renderMathInElement(assistantBody);

      // Append Sources Accordion if excerpts exist
      if (contextExcerpts.length > 0) {
        const sourcesHtml = buildSourcesAccordion(contextExcerpts, userPrompt);
        assistantBody.insertAdjacentHTML('beforeend', sourcesHtml);
      }
    }

    // Save to active chat history
    saveChatMessage('user', userPrompt);
    saveChatMessage('assistant', fullResponseText, contextExcerpts);

  } catch (err) {
    if (err.name === 'AbortError') {
      assistantBody.innerHTML += '<div class="errbox">Generation stopped by user.</div>';
    } else {
      assistantBody.innerHTML = `<div class="errbox"><strong>Error:</strong> ${esc(err.message)}</div>`;
    }
  } finally {
    isBusy = false;
    abortController = null;
    $('sendBtn').disabled = false;
    $('sendBtn').textContent = '➤';
    setTimeout(() => {
      if (!isBusy) $('tpsWrap').textContent = '';
    }, 4000);
  }
}

// Markdown renderer with clickable citations
function renderMarkdown(text) {
  let html = marked.parse(text);
  // Turn [1], [2] into anchor citations
  html = html.replace(/\[(\d+)\]/g, '<a class="cite" href="#src-$1">[$1]</a>');
  return html;
}

// KaTeX auto-renderer wrapper
function renderMathInElement(el) {
  if (window.renderMathInElement) {
    try {
      window.renderMathInElement(el, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false }
        ],
        throwOnError: false
      });
    } catch (e) {
      console.warn('KaTeX render error:', e);
    }
  }
}

// Expandable Sources Accordion
function buildSourcesAccordion(excerpts, query) {
  const qTokens = tokenize(query);

  let html = `<details class="sources" open>
    <summary>Sources (${excerpts.length} verified excerpts from selected documents)</summary>`;

  excerpts.forEach((c, idx) => {
    const srcNum = idx + 1;
    // Highlight matched query terms
    let excerptText = esc(c.text);
    for (const t of qTokens.slice(0, 10)) {
      if (t.length > 2) {
        const re = new RegExp(`(${t})`, 'gi');
        excerptText = excerptText.replace(re, '<mark>$1</mark>');
      }
    }

    html += `
      <div class="src" id="src-${srcNum}">
        <div class="head">
          <span>[${srcNum}] ${esc(c.filename)} — ${esc(c.clause || 'General')}${c.page ? ` (p. ${c.page})` : ''}</span>
          <button onclick="copyExcerptText(this, ${JSON.stringify(c.text)})">Copy</button>
        </div>
        <div class="txt">${excerptText}</div>
      </div>
    `;
  });

  html += '</details>';
  return html;
}

window.copyExcerptText = (btn, text) => {
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = orig; }, 1500);
  });
};

function addMessageToUI(role, innerHtml) {
  const welcome = $('welcome');
  if (welcome) welcome.remove();

  const chatContainer = $('chat');
  const msgDiv = document.createElement('div');
  msgDiv.className = 'msg ' + role;
  msgDiv.innerHTML = `
    <div class="avatar">${role === 'user' ? 'You' : 'AI'}</div>
    <div class="body">${innerHtml}</div>
  `;
  chatContainer.appendChild(msgDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return msgDiv.querySelector('.body');
}

// ----------------------------------------------------
// Chat Sessions Management (LocalStorage)
// ----------------------------------------------------
function loadChats() {
  try {
    const saved = localStorage.getItem('researcher_chats');
    if (saved) {
      chats = JSON.parse(saved);
    }
  } catch (e) {
    console.error('Failed to load chats:', e);
    chats = [];
  }
  if (!chats || chats.length === 0) {
    createNewChat();
  } else {
    activeChatId = chats[0].id;
    renderChatList();
    renderActiveChatMessages();
  }
}

function saveChatsToStorage() {
  try {
    localStorage.setItem('researcher_chats', JSON.stringify(chats));
  } catch (e) {
    console.error('Failed to save chats:', e);
  }
}

function createNewChat() {
  const newId = 'chat_' + Date.now();
  const chatObj = {
    id: newId,
    title: 'New conversation',
    messages: [],
    created_at: new Date().toISOString()
  };
  chats.unshift(chatObj);
  activeChatId = newId;
  saveChatsToStorage();
  renderChatList();
  renderActiveChatMessages();
}

function getActiveChat() {
  return chats.find(c => c.id === activeChatId) || chats[0];
}

function saveChatMessage(role, content, sources = []) {
  const currentChat = getActiveChat();
  if (!currentChat) return;

  currentChat.messages.push({
    role,
    content,
    sources,
    timestamp: new Date().toISOString()
  });

  // Set chat title from first user query
  if (currentChat.messages.length === 2 && currentChat.title === 'New conversation') {
    const firstUserMsg = currentChat.messages.find(m => m.role === 'user');
    if (firstUserMsg) {
      currentChat.title = firstUserMsg.content.slice(0, 32) + (firstUserMsg.content.length > 32 ? '…' : '');
    }
  }

  saveChatsToStorage();
  renderChatList();
}

function renderChatList() {
  const container = $('chatList');
  container.innerHTML = '';

  for (const c of chats) {
    const item = document.createElement('div');
    item.className = 'chatItem' + (c.id === activeChatId ? ' active' : '');
    item.onclick = () => switchChat(c.id);
    item.innerHTML = `
      <span class="t" title="${esc(c.title)}">${esc(c.title)}</span>
      <button title="Delete conversation" onclick="event.stopPropagation(); deleteChat('${c.id}')">✕</button>
    `;
    container.appendChild(item);
  }
}

function switchChat(chatId) {
  if (isBusy) {
    alert('Please wait for the current response to finish or stop generation.');
    return;
  }
  activeChatId = chatId;
  renderChatList();
  renderActiveChatMessages();
}

function deleteChat(chatId) {
  if (!confirm('Delete this conversation?')) return;
  chats = chats.filter(c => c.id !== chatId);
  if (chats.length === 0) {
    createNewChat();
  } else {
    if (activeChatId === chatId) {
      activeChatId = chats[0].id;
    }
    saveChatsToStorage();
    renderChatList();
    renderActiveChatMessages();
  }
}

function renderActiveChatMessages() {
  const chatContainer = $('chat');
  chatContainer.innerHTML = '';

  const currentChat = getActiveChat();
  if (!currentChat || currentChat.messages.length === 0) {
    chatContainer.innerHTML = `
      <div id="welcome">
        <h2>Chat with your codes &amp; standards</h2>
        <p>Upload documents on the left, then ask anything. Answers cite the exact source and page.</p>
        <div class="hint" onclick="useHint(this)">What is the design factor for restrained pipelines?</div>
        <div class="hint" onclick="useHint(this)">Summarize the hydrotest requirements across my documents.</div>
        <div class="hint" onclick="useHint(this)">Which clause covers welding qualification for duplex stainless?</div>
      </div>
    `;
    return;
  }

  for (const m of currentChat.messages) {
    let bodyHtml = renderMarkdown(m.content);
    if (m.sources && m.sources.length > 0) {
      bodyHtml += buildSourcesAccordion(m.sources, '');
    }
    const msgDiv = document.createElement('div');
    msgDiv.className = 'msg ' + m.role;
    msgDiv.innerHTML = `
      <div class="avatar">${m.role === 'user' ? 'You' : 'AI'}</div>
      <div class="body">${bodyHtml}</div>
    `;
    chatContainer.appendChild(msgDiv);
    renderMathInElement(msgDiv);
  }

  chatContainer.scrollTop = chatContainer.scrollHeight;
}

window.useHint = el => {
  $('input').value = el.textContent.trim();
  $('input').focus();
};

// ----------------------------------------------------
// UI Controls & Settings Modal
// ----------------------------------------------------
function openSettingsModal() {
  $('s_key').value = settings.apiKey || '';
  $('s_model').value = settings.model || 'gemini-3-flash-preview';
  $('s_topk').value = settings.topK || 6;
  $('s_vw').value = settings.vectorWeight || 0.5;
  $('chatTest').textContent = '';
  $('modalBg').classList.add('open');
}

function closeSettingsModal() {
  $('modalBg').classList.remove('open');
}

async function testGeminiConnection() {
  const key = $('s_key').value.trim();
  const testLine = $('chatTest');
  testLine.className = 'testline';
  testLine.textContent = 'Testing connection & fetching models…';

  if (!key) {
    testLine.className = 'testline bad';
    testLine.textContent = 'Error: Please enter an API key.';
    return;
  }

  try {
    const resp = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${key}`);
    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.error?.message || `HTTP ${resp.status}`);
    }

    testLine.className = 'testline ok';
    testLine.textContent = 'Connected! API Key is valid.';
    $('chatDot').className = 'dot ok';
    $('chatStatus').textContent = 'AI connected';

    // Populate model datalist & select
    const datalist = $('modelList');
    const select = $('s_modelSelect');
    datalist.innerHTML = '';
    select.innerHTML = '<option value="">Select fetched model…</option>';

    const geminiModels = (data.models || []).filter(m => m.supportedGenerationMethods?.includes('generateContent'));
    for (const m of geminiModels) {
      const cleanName = m.name.replace('models/', '');
      const opt = document.createElement('option');
      opt.value = cleanName;
      datalist.appendChild(opt);

      const selOpt = document.createElement('option');
      selOpt.value = cleanName;
      selOpt.textContent = cleanName;
      select.appendChild(selOpt);
    }
  } catch (err) {
    testLine.className = 'testline bad';
    testLine.textContent = 'Connection failed: ' + err.message;
    $('chatDot').className = 'dot bad';
    $('chatStatus').textContent = 'AI offline — check Settings';
  }
}

// ----------------------------------------------------
// Event Listeners & Bootstrapping
// ----------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  loadSettings();
  loadChats();
  initializeLibrary();

  // Status check
  if (settings.apiKey) {
    $('chatDot').className = 'dot ok';
    $('chatStatus').textContent = 'AI connected';
  } else {
    $('chatDot').className = 'dot bad';
    $('chatStatus').textContent = 'AI offline — click Settings';
  }

  // Gear & Settings Modal
  $('gearBtn').onclick = openSettingsModal;
  $('cancelBtn').onclick = closeSettingsModal;
  $('saveBtn').onclick = () => {
    settings.apiKey = $('s_key').value.trim();
    settings.model = $('s_model').value.trim() || 'gemini-3-flash-preview';
    settings.topK = parseInt($('s_topk').value, 10) || 6;
    settings.vectorWeight = parseFloat($('s_vw').value) || 0.5;
    saveSettings();
    closeSettingsModal();
    if (settings.apiKey) {
      $('chatDot').className = 'dot ok';
      $('chatStatus').textContent = 'AI connected';
    }
  };

  $('testChatBtn').onclick = testGeminiConnection;
  $('s_modelSelect').onchange = () => {
    if ($('s_modelSelect').value) {
      $('s_model').value = $('s_modelSelect').value;
    }
  };

  // Chat Actions
  $('newChat').onclick = createNewChat;

  // Selection Bar Actions
  $('selAll').onclick = () => selectAll(true);
  $('selNone').onclick = () => selectAll(false);
  $('clearAll').onclick = clearLocalDocs;

  // Mobile Drawer
  const mobileMenuBtn = $('mobileMenuBtn');
  const sidebar = $('sidebar');
  const overlay = $('mobileOverlay');

  mobileMenuBtn.onclick = () => {
    sidebar.classList.toggle('open');
    overlay.classList.toggle('open');
  };
  overlay.onclick = () => {
    sidebar.classList.remove('open');
    overlay.classList.remove('open');
  };

  // Send & Input handling
  const inputEl = $('input');
  const sendBtn = $('sendBtn');

  const handleSend = () => {
    if (isBusy) {
      if (abortController) abortController.abort();
      return;
    }
    const val = inputEl.value.trim();
    if (!val) return;
    streamChatResponse(val);
  };

  sendBtn.onclick = handleSend;

  inputEl.onkeydown = e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  inputEl.oninput = () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
  };
});
