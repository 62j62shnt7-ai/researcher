/**
 * Researcher AI — Client-Side Web Application
 * Matches the exact design, sidebar workflow, document selection, and chat experience of the local app.
 */

(() => {
  'use strict';

  // --- Utility Selector ---
  const $ = (id) => document.getElementById(id);
  function esc(s) {
    return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  // --- Application State ---
  const state = {
    apiKey: localStorage.getItem('researcher_gemini_key') || '',
    model: localStorage.getItem('researcher_gemini_model') || 'gemini-3-flash-preview',
    topK: parseInt(localStorage.getItem('researcher_top_k') || '8', 10),
    vectorWeight: parseFloat(localStorage.getItem('researcher_vector_weight') || '0.6'),
    repoDocs: [],      // From knowledge_base.json
    repoChunks: [],    // From knowledge_base.json
    localDocs: [],     // From IndexedDB
    localChunks: [],   // From IndexedDB
    docSel: {},        // { [docKey]: boolean }
    chats: [],         // [{ id, title, messages, updatedAt }]
    currentChatId: null,
    busy: false,
    aborter: null,
    lastQuestion: ''
  };

  // --- IndexedDB for Uploaded Documents ---
  const DB_NAME = 'ResearcherWebDB';
  const DB_VERSION = 1;
  let dbPromise = null;

  function getDB() {
    if (!dbPromise) {
      dbPromise = new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = (e) => {
          const db = e.target.result;
          if (!db.objectStoreNames.contains('documents')) {
            db.createObjectStore('documents', { keyPath: 'id' });
          }
          if (!db.objectStoreNames.contains('chunks')) {
            const cs = db.createObjectStore('chunks', { keyPath: 'id' });
            cs.createIndex('docId', 'docId', { unique: false });
          }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
    }
    return dbPromise;
  }

  async function loadLocalData() {
    try {
      const db = await getDB();
      return new Promise((resolve) => {
        const tx = db.transaction(['documents', 'chunks'], 'readonly');
        const docsReq = tx.objectStore('documents').getAll();
        const chunksReq = tx.objectStore('chunks').getAll();
        tx.oncomplete = () => {
          state.localDocs = docsReq.result || [];
          state.localChunks = chunksReq.result || [];
          resolve();
        };
        tx.onerror = () => resolve();
      });
    } catch {
      state.localDocs = [];
      state.localChunks = [];
    }
  }

  async function saveLocalDoc(docMeta, chunks) {
    const db = await getDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(['documents', 'chunks'], 'readwrite');
      tx.objectStore('documents').put(docMeta);
      const cStore = tx.objectStore('chunks');
      chunks.forEach((c) => cStore.put(c));
      tx.oncomplete = async () => {
        await loadLocalData();
        resolve();
      };
      tx.onerror = () => reject(tx.error);
    });
  }

  async function deleteLocalDoc(docId) {
    const db = await getDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(['documents', 'chunks'], 'readwrite');
      tx.objectStore('documents').delete(docId);
      const cStore = tx.objectStore('chunks');
      const idx = cStore.index('docId');
      const req = idx.getAllKeys(docId);
      req.onsuccess = () => {
        (req.result || []).forEach((k) => cStore.delete(k));
      };
      tx.oncomplete = async () => {
        await loadLocalData();
        resolve();
      };
      tx.onerror = () => reject(tx.error);
    });
  }

  async function clearAllLocalDocs() {
    const db = await getDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(['documents', 'chunks'], 'readwrite');
      tx.objectStore('documents').clear();
      tx.objectStore('chunks').clear();
      tx.oncomplete = async () => {
        await loadLocalData();
        resolve();
      };
      tx.onerror = () => reject(tx.error);
    });
  }

  // --- Load Pre-indexed Repository Knowledge Base ---
  async function loadRepoKB() {
    try {
      const res = await fetch('knowledge_base.json', { cache: 'no-cache' });
      if (res.ok) {
        const data = await res.json();
        state.repoDocs = data.documents || [];
        state.repoChunks = data.chunks || [];
      }
    } catch (e) {
      console.warn('No pre-indexed knowledge_base.json found:', e);
      state.repoDocs = [];
      state.repoChunks = [];
    }
  }

  // --- Tokenizer & BM25 Scoring ---
  function tokenize(text) {
    if (!text) return [];
    return text.toLowerCase().match(/[a-z0-9]+(?:[\.\-_:][a-z0-9]+)*/gi) || [];
  }

  function scoreBM25(queryTokens, chunkTokens) {
    if (!queryTokens.length || !chunkTokens.length) return 0;
    const chunkTokenMap = {};
    chunkTokens.forEach((t) => (chunkTokenMap[t] = (chunkTokenMap[t] || 0) + 1));
    let score = 0;
    queryTokens.forEach((qt) => {
      if (chunkTokenMap[qt]) {
        const weight = qt.includes('.') || qt.includes('-') || qt.length > 5 ? 3.0 : 1.5;
        score += chunkTokenMap[qt] * weight;
      }
    });
    return score;
  }

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

  // --- Hybrid Retrieval across Selected Documents ---
  function retrieve(question, topK = 8) {
    const qTokens = tokenize(question);

    // Combine chunks from checked documents
    let activeChunks = [];

    // 1. Repo chunks
    state.repoChunks.forEach((c) => {
      const key = c.file || 'README.md';
      if (state.docSel[key] !== false) {
        activeChunks.push(c);
      }
    });

    // 2. Local uploaded chunks
    state.localChunks.forEach((c) => {
      const key = c.docId || c.file;
      if (state.docSel[key] !== false) {
        activeChunks.push(c);
      }
    });

    if (activeChunks.length === 0) {
      activeChunks = [...state.repoChunks, ...state.localChunks];
    }

    const scored = activeChunks.map((chunk) => {
      const bm25 = scoreBM25(qTokens, chunk.tokens || tokenize(chunk.text));
      return { chunk, score: bm25 };
    });

    scored.sort((a, b) => b.score - a.score);

    let hits = scored.slice(0, topK).filter((s) => s.score > 0.05).map((s) => ({
      filename: s.chunk.file || 'document',
      location: s.chunk.clause || (s.chunk.page ? `p. ${s.chunk.page}` : ''),
      text: s.chunk.text,
      score: s.score
    }));

    // Fallback: If broad question produced 0 keyword matches, provide initial document chunks
    if (hits.length === 0 && activeChunks.length > 0) {
      hits = activeChunks.slice(0, Math.min(topK, 5)).map((c) => ({
        filename: c.file || 'document',
        location: c.clause || (c.page ? `p. ${c.page}` : ''),
        text: c.text,
        score: 0.1
      }));
    }

    return hits;
  }

  // --- Document Parsing ---
  async function parsePdf(file) {
    const buf = await file.arrayBuffer();
    const pages = [];
    if (window.pdfjsLib) {
      pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
      const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
      for (let i = 1; i <= pdf.numPages; i++) {
        try {
          const page = await pdf.getPage(i);
          const tc = await page.getTextContent();
          const t = tc.items.map((it) => it.str).join(' ').trim();
          if (t) pages.push({ page: i, text: t });
        } catch {}
      }
    }
    return pages;
  }

  async function parseDocx(file) {
    const buf = await file.arrayBuffer();
    if (window.mammoth) {
      const res = await mammoth.extractRawText({ arrayBuffer: buf });
      const full = res.value || '';
      const pages = [];
      const chunkSize = 2000;
      let p = 1;
      for (let i = 0; i < full.length; i += chunkSize) {
        pages.push({ page: p++, text: full.slice(i, i + chunkSize) });
      }
      return pages;
    }
    return [];
  }

  async function ingestFiles(files) {
    if (!files || !files.length) return;
    for (const file of files) {
      const filename = file.name;
      const ext = filename.split('.').pop().toLowerCase();
      let pages = [];
      try {
        if (ext === 'pdf') pages = await parsePdf(file);
        else if (ext === 'docx' || ext === 'doc') pages = await parseDocx(file);
        else {
          const t = await file.text();
          pages = [{ page: 1, text: t }];
        }
      } catch (e) {
        console.warn('Parsing failed for', filename, e);
        continue;
      }

      const docId = `local_${Date.now()}_${filename.replace(/[^a-zA-Z0-9._-]/g, '_')}`;
      const chunks = [];
      let cId = 0;
      const chunkSize = 1800;
      const overlap = 200;

      for (const p of pages) {
        let start = 0;
        const text = p.text || '';
        while (start < text.length) {
          const cStr = text.slice(start, start + chunkSize).trim();
          if (cStr) {
            const clauseMatch = cStr.match(/(?:(?:para|section|clause|article|part)\s+[\d\.]+|[\d]+\.[\d]+(?:\.[\d]+)?)/i);
            chunks.push({
              id: `${docId}_c${cId++}`,
              docId,
              file: filename,
              page: p.page,
              clause: clauseMatch ? clauseMatch[0] : `p. ${p.page}`,
              text: cStr,
              tokens: tokenize(cStr),
              embedding: null
            });
          }
          start += (chunkSize - overlap);
        }
      }

      const meta = {
        id: docId,
        filename,
        page_count: pages.length,
        chunk_count: chunks.length,
        isLocal: true,
        addedAt: Date.now()
      };

      state.docSel[docId] = true;
      await saveLocalDoc(meta, chunks);
    }
    renderDocs();
  }

  // --- Documents Sidebar UI ---
  function getAllDocEntries() {
    const list = [];
    state.repoDocs.forEach((d) => {
      const key = d.filename;
      if (state.docSel[key] === undefined) state.docSel[key] = true;
      list.push({
        id: key,
        filename: d.filename,
        pages: d.page_count,
        chunks: d.chunk_count,
        isLocal: false
      });
    });
    state.localDocs.forEach((d) => {
      const key = d.id;
      if (state.docSel[key] === undefined) state.docSel[key] = true;
      list.push({
        id: key,
        filename: d.filename,
        pages: d.page_count,
        chunks: d.chunk_count,
        isLocal: true
      });
    });
    return list;
  }

  function renderDocs() {
    const listEl = $('docList');
    listEl.innerHTML = '';
    const docs = getAllDocEntries();

    if (docs.length === 0) {
      listEl.innerHTML = '<div style="font-size:12px;color:var(--muted);padding:12px 6px;">No documents in library yet. Drop PDF/Word files above or click Upload.</div>';
    }

    docs.forEach((d) => {
      const div = document.createElement('div');
      div.className = 'doc' + (state.docSel[d.id] ? '' : ' excluded');
      const badgeText = d.isLocal ? 'Local' : 'Repo';
      const badgeCls = d.isLocal ? 'b-ready' : 'b-busy';

      div.innerHTML = `
        <input type="checkbox" ${state.docSel[d.id] ? 'checked' : ''} title="Include in chat">
        <div style="flex:1;min-width:0">
          <div class="name" title="${esc(d.filename)}">${esc(d.filename)}</div>
          <div class="meta">${d.pages ? d.pages + ' pages • ' : ''}${d.chunks || 0} chunks</div>
        </div>
        <span class="badge ${badgeCls}">${badgeText}</span>
        ${d.isLocal ? `<button title="Delete document" class="btn-del-doc">✕</button>` : ''}
      `;

      div.querySelector('input').onchange = (e) => {
        state.docSel[d.id] = e.target.checked;
        div.classList.toggle('excluded', !e.target.checked);
        updateSelInfo();
      };

      const delBtn = div.querySelector('.btn-del-doc');
      if (delBtn) {
        delBtn.onclick = async (e) => {
          e.stopPropagation();
          if (confirm(`Remove "${d.filename}" from library?`)) {
            await deleteLocalDoc(d.id);
            delete state.docSel[d.id];
            renderDocs();
          }
        };
      }

      listEl.appendChild(div);
    });

    updateSelInfo();
    updateStats();
  }

  function updateSelInfo() {
    const docs = getAllDocEntries();
    const checkedCount = docs.filter((d) => state.docSel[d.id] !== false).length;
    $('selInfo').textContent = `${checkedCount} of ${docs.length} selected`;
  }

  function updateStats() {
    const docs = getAllDocEntries();
    const totalChunks = docs.reduce((a, d) => a + (d.chunks || 0), 0);
    $('stats').textContent = `${docs.length} document${docs.length === 1 ? '' : 's'} · ${totalChunks} chunks`;
  }

  // --- Chat History Persistence ---
  function loadSavedChats() {
    try {
      state.chats = JSON.parse(localStorage.getItem('researcher_chats') || '[]');
    } catch {
      state.chats = [];
    }
  }

  function saveChats() {
    localStorage.setItem('researcher_chats', JSON.stringify(state.chats.slice(0, 30)));
  }

  function renderChatList() {
    const el = $('chatList');
    el.innerHTML = '';
    if (state.chats.length === 0) {
      el.innerHTML = '<div style="font-size:12px;color:var(--muted);padding:6px 8px;">No saved chats yet</div>';
      return;
    }

    state.chats.forEach((ch) => {
      const div = document.createElement('div');
      div.className = 'chatItem' + (ch.id === state.currentChatId ? ' active' : '');
      div.innerHTML = `
        <span class="t" title="${esc(ch.title)} — double click to rename">${esc(ch.title)}</span>
        <button title="Delete chat" class="delChatBtn">✕</button>
      `;

      div.onclick = () => selectChat(ch.id);

      div.ondblclick = () => {
        const newTitle = prompt('Rename chat:', ch.title);
        if (newTitle && newTitle.trim()) {
          ch.title = newTitle.trim();
          saveChats();
          renderChatList();
        }
      };

      div.querySelector('.delChatBtn').onclick = (e) => {
        e.stopPropagation();
        if (confirm('Delete this chat?')) {
          state.chats = state.chats.filter((c) => c.id !== ch.id);
          if (state.currentChatId === ch.id) {
            state.currentChatId = null;
            $('chat').innerHTML = '';
            showWelcome();
          }
          saveChats();
          renderChatList();
        }
      };

      el.appendChild(div);
    });
  }

  function selectChat(id) {
    if (state.busy) state.aborter?.abort();
    const ch = state.chats.find((c) => c.id === id);
    if (!ch) return;
    state.currentChatId = id;
    $('chat').innerHTML = '';
    ch.messages.forEach((m) => {
      if (m.role === 'user') {
        state.lastQuestion = m.content;
        addMsg('user', esc(m.content).replace(/\n/g, '<br>'));
      } else {
        const b = addMsg('assistant', marked.parse(m.content));
        if (m.sources && m.sources.length) {
          renderSources(b, m.sources);
        }
      }
    });
    renderChatList();
    closeMobileSidebar();
  }

  function showWelcome() {
    $('chat').innerHTML = `
      <div id="welcome">
        <h2>Chat with your codes &amp; standards</h2>
        <p>Select documents on the left, then ask anything. Answers cite the exact standard clause and page.</p>
        <div class="hint" data-prompt="What is the design factor for restrained pipelines according to ASME standards?">What is the design factor for restrained pipelines?</div>
        <div class="hint" data-prompt="Summarize the repair and hydrotest requirements across my documents.">Summarize the repair and hydrotest requirements across my documents.</div>
        <div class="hint" data-prompt="Which clause covers welded split sleeve repairs and what are the limitations?">Which clause covers welded split sleeve repairs and what are the limitations?</div>
      </div>
    `;
    attachHints();
  }

  function attachHints() {
    document.querySelectorAll('#welcome .hint').forEach((el) => {
      el.onclick = () => {
        $('input').value = el.getAttribute('data-prompt');
        $('input').focus();
        autoGrow();
      };
    });
  }

  // --- Messages & Citations Rendering ---
  function addMsg(role, html) {
    $('welcome')?.remove();
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    div.innerHTML = `<div class="avatar">${role === 'user' ? 'You' : 'AI'}</div><div class="body">${html}</div>`;
    $('chat').appendChild(div);
    $('chat').scrollTop = $('chat').scrollHeight;
    return div.querySelector('.body');
  }

  function hlEsc(text) {
    let h = esc(text);
    const toks = [...new Set(((state.lastQuestion || '').match(/[A-Za-z0-9_.\-]{3,}/g) || []).map((t) => t.toLowerCase()))];
    for (const t of toks.slice(0, 10)) {
      const re = new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
      h = h.replace(re, '<mark>$1</mark>');
    }
    return h;
  }

  function renderSources(container, sources) {
    if (!sources || !sources.length) return;
    const det = document.createElement('details');
    det.className = 'sources';
    let html = `<summary>📄 ${sources.length} source excerpt${sources.length > 1 ? 's' : ''}</summary>`;
    sources.forEach((s, idx) => {
      const head = `[${idx + 1}] ${esc(s.filename)}${s.location ? ' — ' + esc(s.location) : ''}`;
      html += `<div class="src"><div class="head"><span>${head}</span></div><div class="txt">${hlEsc(s.text)}</div></div>`;
    });
    det.innerHTML = html;
    container.appendChild(det);
  }

  function renderMath(container) {
    if (window.renderMathInElement) {
      try {
        renderMathInElement(container, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\[', right: '\\]', display: true },
            { left: '\\(', right: '\\)', display: false }
          ],
          throwOnError: false
        });
      } catch {}
    }
  }

  // --- Chat Streaming with Gemini 3 ---
  async function send() {
    const text = $('input').value.trim();
    if (!text || state.busy) return;

    if (!state.apiKey) {
      openSettings();
      alert('Please enter your Google Gemini API Key in Settings to start chatting.');
      return;
    }

    state.busy = true;
    state.lastQuestion = text;
    $('sendBtn').textContent = '■';
    $('sendBtn').title = 'Stop generating';
    $('input').value = '';
    autoGrow();

    addMsg('user', esc(text).replace(/\n/g, '<br>'));
    const bodyEl = addMsg('assistant', '<span class="spinner"></span> Searching library &amp; matching clauses…');

    const useLibrary = $('useLibrary').checked;
    const isCompare = $('compareMode').checked;
    let sources = [];

    if (useLibrary) {
      sources = retrieve(text, state.topK);
    }

    // Build Prompt Contents
    let contextText = '';
    if (sources && sources.length > 0) {
      contextText = 'ENGINEERING CODE & STANDARD EXCERPTS (Ground your answer strictly on these excerpts):\n\n' +
        sources.map((s, i) => `[Source ${i + 1}]: Standard: "${s.filename}", Clause: "${s.location}"\nExcerpt:\n${s.text}`).join('\n\n---\n\n');
    }

    const compareInstruction = isCompare
      ? '\n\nCOMPARISON MODE: You are comparing requirements across documents. State key differences, contrast values/tolerances in a comparison table, and note which requirement is more stringent.'
      : '';

    const systemInstruction = `You are Researcher AI, an expert engineering codes & standards assistant.
Answer the user's question directly, clearly, and logically using the provided code excerpts.
Explain requirements practically: what they mean, conditions, tolerances, and design limits.
Always cite the exact clause numbers (e.g. Para 304.1.2, Article 202, UG-27), values, formulas, and source brackets like [Source 1] or [1].
Never output raw internal thinking, prompt instructions, or ungrounded guesses.${compareInstruction}`;

    const modelsToTry = [state.model, 'gemini-3-flash-preview', 'gemini-3.7-flash', 'gemini-3.8-flash', 'gemma-4-31b-it'].filter(
      (m, i, arr) => m && arr.indexOf(m) === i && !m.includes('2.5')
    );

    let fullAnswer = '';
    let nTok = 0;
    const tokStart = performance.now();
    $('tpsWrap').textContent = '';

    const contents = [
      {
        role: 'user',
        parts: [{ text: contextText ? `${contextText}\n\nUSER QUESTION: ${text}` : text }]
      }
    ];

    state.aborter = new AbortController();

    try {
      let success = false;
      for (const modelName of modelsToTry) {
        if (state.aborter.signal.aborted) break;
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:streamGenerateContent?alt=sse&key=${state.apiKey}`;

        try {
          const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              contents,
              systemInstruction: { parts: [{ text: systemInstruction }] },
              generationConfig: { temperature: 0.2, maxOutputTokens: 8192 }
            }),
            signal: state.aborter.signal
          });

          if (!res.ok) continue;

          bodyEl.innerHTML = '';
          const reader = res.body.getReader();
          const decoder = new TextDecoder('utf-8');
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const jsonStr = line.slice(6).trim();
                if (!jsonStr) continue;
                try {
                  const data = JSON.parse(jsonStr);
                  const parts = data.candidates?.[0]?.content?.parts || [];
                  for (const p of parts) {
                    if (p.thought) continue;
                    let t = p.text || '';
                    t = t.replace(/<think>[\s\S]*?(<\/think>|$)/g, '').replace(/<thought>[\s\S]*?(<\/thought>|$)/g, '');
                    if (t) {
                      fullAnswer += t;
                      nTok += Math.max(1, Math.round(t.length / 4));
                      bodyEl.innerHTML = marked.parse(fullAnswer);
                      const elapsed = (performance.now() - tokStart) / 1000;
                      $('tpsWrap').textContent = `⚡ ${(nTok / Math.max(elapsed, 0.001)).toFixed(1)} tok/s avg · ${nTok} tokens`;
                      $('chat').scrollTop = $('chat').scrollHeight;
                    }
                  }
                } catch {}
              }
            }
          }

          if (fullAnswer.trim()) {
            success = true;
            break;
          }
        } catch (callErr) {
          if (callErr.name === 'AbortError') break;
        }
      }

      if (!success && !state.aborter.signal.aborted) {
        bodyEl.innerHTML = '<span style="color:var(--bad)">Failed to get response. Please check your Gemini API key in Settings.</span>';
      } else {
        renderMath(bodyEl);
        if (sources.length) {
          renderSources(bodyEl, sources);
        }

        // Save conversation
        if (!state.currentChatId) {
          state.currentChatId = 'chat_' + Date.now();
          state.chats.unshift({
            id: state.currentChatId,
            title: text.slice(0, 50),
            messages: [],
            updatedAt: Date.now()
          });
        }
        const activeCh = state.chats.find((c) => c.id === state.currentChatId);
        if (activeCh) {
          activeCh.messages.push({ role: 'user', content: text });
          activeCh.messages.push({ role: 'assistant', content: fullAnswer, sources });
          activeCh.updatedAt = Date.now();
          saveChats();
          renderChatList();
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        bodyEl.innerHTML = `<span style="color:var(--bad)">Error: ${esc(err.message)}</span>`;
      }
    } finally {
      state.busy = false;
      state.aborter = null;
      $('sendBtn').textContent = '➤';
      $('sendBtn').title = 'Send';
      $('chat').scrollTop = $('chat').scrollHeight;
    }
  }

  function autoGrow() {
    const i = $('input');
    i.style.height = 'auto';
    i.style.height = Math.min(i.scrollHeight, 160) + 'px';
  }

  // --- Status & Settings ---
  function updateConnectionStatus() {
    if (state.apiKey) {
      $('chatDot').className = 'dot ok';
      $('chatStatus').textContent = `AI connected (${state.model})`;
    } else {
      $('chatDot').className = 'dot bad';
      $('chatStatus').textContent = 'AI offline — set API key';
    }
    const totalDocs = state.repoDocs.length + state.localDocs.length;
    if (totalDocs > 0) {
      $('embDot').className = 'dot ok';
      $('embStatus').textContent = `library ready (${totalDocs} doc${totalDocs === 1 ? '' : 's'})`;
    } else {
      $('embDot').className = 'dot';
      $('embStatus').textContent = 'no docs loaded';
    }
  }

  function openSettings() {
    $('s_key').value = state.apiKey;
    $('s_model').value = state.model;
    $('s_topk').value = state.topK;
    $('s_vw').value = state.vectorWeight;
    $('chatTest').textContent = '';
    $('modalBg').classList.add('open');
  }

  function closeSettings() {
    $('modalBg').classList.remove('open');
  }

  async function testAndFetchModels() {
    const key = $('s_key').value.trim();
    if (!key) {
      $('chatTest').textContent = '✗ API key required';
      $('chatTest').className = 'testline bad';
      return;
    }
    $('chatTest').textContent = 'Connecting to Google Gemini API…';
    $('chatTest').className = 'testline';

    try {
      const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${key}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const models = (data.models || [])
        .map((m) => (m.name || '').replace('models/', ''))
        .filter((m) => m.includes('flash') || m.includes('pro') || m.includes('gemma'));

      const sel = $('s_modelSelect');
      const datalist = $('modelList');
      datalist.innerHTML = models.map((m) => `<option value="${esc(m)}">`).join('');
      sel.innerHTML = '<option value="">Select model…</option>' + models.map((m) => `<option value="${esc(m)}">${esc(m)}</option>`).join('');

      if (!$('s_model').value && models.length) {
        $('s_model').value = models.includes('gemini-3-flash-preview') ? 'gemini-3-flash-preview' : models[0];
      }

      $('chatTest').textContent = `✓ Connected — ${models.length} Gemini models discovered!`;
      $('chatTest').className = 'testline ok';
    } catch (e) {
      $('chatTest').textContent = '✗ Connection failed: ' + e.message;
      $('chatTest').className = 'testline bad';
    }
  }

  function saveSettings() {
    state.apiKey = $('s_key').value.trim();
    state.model = $('s_model').value.trim() || 'gemini-3-flash-preview';
    state.topK = parseInt($('s_topk').value, 10) || 8;
    state.vectorWeight = parseFloat($('s_vw').value) || 0.6;

    localStorage.setItem('researcher_gemini_key', state.apiKey);
    localStorage.setItem('researcher_gemini_model', state.model);
    localStorage.setItem('researcher_top_k', state.topK.toString());
    localStorage.setItem('researcher_vector_weight', state.vectorWeight.toString());

    updateConnectionStatus();
    closeSettings();
  }

  // --- Mobile Drawer Controls ---
  function toggleMobileSidebar() {
    $('sidebar').classList.toggle('open');
    $('sidebarBackdrop').classList.toggle('open');
  }

  function closeMobileSidebar() {
    $('sidebar').classList.remove('open');
    $('sidebarBackdrop').classList.remove('open');
  }

  // --- Initialization & Event Listeners ---
  async function init() {
    // 1. Load Data
    await loadLocalData();
    await loadRepoKB();
    loadSavedChats();

    // 2. Initial Render
    renderDocs();
    renderChatList();
    updateConnectionStatus();
    showWelcome();

    // 3. Document Drop & Upload Listeners
    const zone = $('uploadZone');
    const fi = $('fileInput');
    zone.onclick = () => fi.click();
    $('uploadBtnSmall').onclick = () => fi.click();

    zone.ondragover = (e) => { e.preventDefault(); zone.classList.add('drag'); };
    zone.ondragleave = () => zone.classList.remove('drag');
    zone.ondrop = (e) => {
      e.preventDefault();
      zone.classList.remove('drag');
      ingestFiles(e.dataTransfer.files);
    };
    fi.onchange = () => {
      ingestFiles(fi.files);
      fi.value = '';
    };

    // 4. Selection Buttons
    $('selAllBtn').onclick = () => {
      getAllDocEntries().forEach((d) => { state.docSel[d.id] = true; });
      renderDocs();
    };
    $('selNoneBtn').onclick = () => {
      getAllDocEntries().forEach((d) => { state.docSel[d.id] = false; });
      renderDocs();
    };
    $('clearAllDocsBtn').onclick = async () => {
      if (confirm('Delete all uploaded local documents from this device?')) {
        await clearAllLocalDocs();
        renderDocs();
      }
    };

    // 5. Chat Controls
    $('sendBtn').onclick = () => {
      if (state.busy) state.aborter?.abort();
      else send();
    };
    $('input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });
    $('input').addEventListener('input', autoGrow);

    $('newChat').onclick = () => {
      state.currentChatId = null;
      $('chat').innerHTML = '';
      showWelcome();
      renderChatList();
    };

    // 6. Settings Modal
    $('gearBtn').onclick = openSettings;
    $('cancelBtn').onclick = closeSettings;
    $('saveBtn').onclick = saveSettings;
    $('testChatBtn').onclick = testAndFetchModels;
    $('modalBg').onclick = (e) => { if (e.target === $('modalBg')) closeSettings(); };
    $('s_modelSelect').onchange = () => {
      if ($('s_modelSelect').value) $('s_model').value = $('s_modelSelect').value;
    };

    // 7. Mobile Navigation
    $('mobileMenuBtn').onclick = toggleMobileSidebar;
    $('sidebarBackdrop').onclick = closeMobileSidebar;
  }

  window.addEventListener('DOMContentLoaded', init);
})();
