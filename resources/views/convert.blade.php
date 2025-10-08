<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ReSeeta – Convert</title>

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">

  <!-- Keep your existing palette & layout -->
  <link rel="stylesheet" href="{{ asset('css/convert.css') }}">
  <meta name="csrf-token" content="{{ csrf_token() }}">

  <style>
    /* Small, tasteful tweaks that reuse your variables from convert.css */
    :root {
      /* relies on --ink, --ink-2, --teal, --shell, --brand already defined */
    }

    /* Page polish */
    .convert-shell {
      box-shadow: 0 20px 40px rgba(0,0,0,.10);
      border: 1px solid #dfe6e4;
    }
    .pane {
      border: 1px solid #e5ecea;
      box-shadow: 0 1px 0 rgba(0,0,0,.03) inset;
    }
    .pane.result {
      border: 1px solid #e5ecea;
      align-items: stretch;
      gap: 10px;
      padding: 16px;
    }
    .pane.result .placeholder {
      font-weight: 700;
      color: var(--ink-2);
      margin: 2px 0 8px;
    }
    .result-box {
      font-size: 15px;
      line-height: 1.55;
      box-shadow: 0 1px 0 rgba(0,0,0,.02) inset;
    }

    /* Actions row refinement */
    .actions {
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
    }
    .actions .group {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      background: #fff;
      border: 1px solid #e5ecea;
      border-radius: 12px;
      box-shadow: 0 1px 0 rgba(0,0,0,.03) inset;
    }
    .actions label {
      font-weight: 600;
      color: var(--ink);
    }
    #modelSelect {
      appearance: none;
      background: #fff;
      border: 1px solid #dfe6e4;
      border-radius: 10px;
      padding: 10px 12px;
      font-weight: 600;
      color: var(--ink-2);
      outline: none;
    }
    #modelSelect:focus {
      border-color: var(--brand);
      box-shadow: 0 0 0 3px rgba(106,177,163,.20);
    }

    /* Toggle – Contextual database (UI only) */
    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      user-select: none;
    }
    .switch {
      --w: 44px;
      --h: 26px;
      position: relative;
      width: var(--w);
      height: var(--h);
      border-radius: var(--h);
      background: #dfe6e4;
      border: 1px solid #cfd8d6;
      transition: background .2s ease, border-color .2s ease;
      cursor: pointer;
    }
    .switch input {
      opacity: 0;
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      margin: 0;
      cursor: pointer;
    }
    .switch .knob {
      position: absolute;
      top: 1px; left: 2px;
      width: calc(var(--h) - 4px);
      height: calc(var(--h) - 4px);
      background: #fff;
      border-radius: 50%;
      box-shadow: 0 1px 2px rgba(0,0,0,.15);
      transition: transform .2s ease;
    }
    .switch input:checked + .knob {
      transform: translateX(calc(var(--w) - var(--h)));
    }
    .switch input:checked ~ .bg {
      background: var(--brand);
      border-color: var(--brand);
    }
    .switch.is-disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .switch.is-disabled input { pointer-events: none; }
    .toggle .label {
      font-weight: 600;
      color: var(--ink-2);
    }

    /* Buttons */
    #startConvert {
      background: var(--ink);
      border: 1px solid #243f42;
    }
    #startConvert:hover {
      transform: translateY(-1px);
    }

    /* Helpers from your existing styles */
    #startConvert:disabled { opacity:.6; cursor:not-allowed; }
    .is-hidden { display:none !important; }
    .model-note { font-size:.9rem; color:#506A6D; }

    /* =========================
       ADD: custom dropdown chevron on #modelSelect
       ========================= */
    /* Ensure vendor appearance resets so native arrow is gone */
    #modelSelect {
      -webkit-appearance: none;
      -moz-appearance: none;
      appearance: none;
      /* Inline SVG chevron, color matches --ink-2 */
      background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'><path d='M1 2l5 5 5-5' stroke='%232A6B6F' stroke-width='2' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>");
      background-repeat: no-repeat;
      background-position: right .55rem center;
      background-size: 12px 8px;
      padding-right: 2rem; /* room for arrow */
    }
    /* In high-contrast or if disabled, dim arrow a bit */
    #modelSelect:disabled {
      background-image: none;
    }
  </style>
</head>
<body>
  <header class="top-header">
    <h1 class="site-title">
      <a href="{{ url('/') }}">ReSeeta</a>
    </h1>
    <nav class="top-nav">
      <a href="{{ url('/') }}" class="{{ Request::is('/') ? 'active' : '' }}">Home</a>
      <a href="{{ url('/about') }}" class="{{ Request::is('about') ? 'active' : '' }}">About</a>
    </nav>
  </header>

  <main class="convert">
    <section class="convert-shell" role="region" aria-label="Prescription recognition">
      <div class="panes">
        <!-- Upload panel -->
        <label class="pane upload" for="fileInput">
          <button type="button"
                  id="btnDeleteUpload"
                  class="icon-btn icon-delete"
                  aria-label="Remove uploaded photo"
                  title="Remove uploaded photo">
            <img src="{{ asset('assets/delete.png') }}" alt="Delete" />
          </button>

          <input id="fileInput" type="file" accept="image/*" hidden>
          <div class="upload-inner">
            <!-- Default state -->
            <div class="upload-icon" aria-hidden="true">Upload Image</div>
            <div class="upload-title">Upload Photo</div>
            <p class="upload-note">
              Maximum file size: 10&nbsp;MB. Only clear, scanned medical prescriptions are accepted.
            </p>

            <!-- Preview band -->
            <img id="previewImage" alt="Image Preview" />

            <!-- Uploading progress -->
            <div class="progress-card" id="progressCard" hidden>
              <div class="progress-title">Uploading</div>

              <div class="progress-row">
                <div class="file-icon" aria-hidden="true">Image</div>
                <div class="file-name" id="fileName">filename.png</div>
                <button class="progress-cancel" id="cancelUpload" type="button" aria-label="Cancel upload">✕</button>
              </div>

              <div class="progress-track" aria-hidden="true">
                <div class="progress-bar" id="progressBar" style="width:0%"></div>
              </div>

              <div class="progress-meta">
                <span id="progressPercent">0%</span>
                <span class="progress-status" id="progressStatus">Uploading…</span>
              </div>
            </div>
          </div>
        </label>

        <!-- Result panel -->
        <div class="pane result" aria-live="polite" aria-atomic="true">
          <button type="button"
                  id="btnHistory"
                  class="icon-btn icon-history"
                  aria-expanded="false"
                  aria-controls="historyPanel"
                  aria-label="View recognition history"
                  title="View history">
            <img src="{{ asset('assets/history.png') }}" alt="History" />
          </button>

          <!-- This text will switch to “Result using MODEL” on success -->
          <span class="placeholder" id="resultHeading">Result Here</span>

          <div class="result-box" id="resultBox"></div>

          <div class="loading" id="convertLoading" hidden aria-live="polite" aria-busy="true">
            <div class="spinner" aria-hidden="true"></div>
            <div class="loading-text">Converting...</div>
          </div>

          <!-- Slide-down History panel -->
          <div id="historyPanel" class="history-panel" hidden>
            <div class="history-header">
              <strong>Recent Results</strong>
              <div>
                <button type="button" id="btnClearHistory" class="history-clear" aria-label="Clear history">Clear</button>
                <button type="button" id="btnCloseHistory" class="history-close" aria-label="Close history">✕</button>
              </div>
            </div>
            <div class="history-body">
              <em>No history yet.</em>
            </div>
          </div>
        </div>
      </div>

      <!-- Actions -->
      <div class="actions">
        <div class="group">
          <label for="modelSelect">Model:</label>
          <select id="modelSelect">
            <option value="vit" selected>ViT-CRNN (Proposed)</option>
            <option value="crnn">CRNN only (Baseline)</option>
          </select>
        </div>

        <div class="group toggle" title="Contextual database (coming soon)">
          <span class="label">Contextual database</span>
          <label class="switch">
            <input type="checkbox" id="contextToggle" />
            <span class="knob"></span>
            <span class="bg" aria-hidden="true"></span>
          </label>
        </div>

        <button id="startConvert" type="button" disabled>Recognize Prescription</button>

        <!-- Subtle note about last model used (optional) -->
        <span id="modelUsedNote" class="model-note" aria-live="polite"></span>
      </div>
    </section>
  </main>

  <footer>
    <p>© {{ date('Y') }} ReSeeta. All Rights Reserved.</p>
  </footer>

<script>
  /* =========================
     Config
  ========================= */
  const API_URL = "{{ route('ocr.predict') }}";

  /* =========================
     Element refs
  ========================= */
  const fileInput = document.getElementById('fileInput');
  const previewImage = document.getElementById('previewImage');

  const startBtn = document.getElementById('startConvert');
  const progressCard = document.getElementById('progressCard');
  const progressBar = document.getElementById('progressBar');
  const progressPercent = document.getElementById('progressPercent');
  const progressStatus = document.getElementById('progressStatus');
  const cancelBtn = document.getElementById('cancelUpload');
  const uploadInner = document.querySelector('.upload-inner');

  const convertLoading = document.getElementById('convertLoading');
  const resultHeading = document.getElementById('resultHeading');
  const resultBox = document.getElementById('resultBox');

  const btnDeleteUpload = document.getElementById('btnDeleteUpload');
  const btnHistory = document.getElementById('btnHistory');
  const btnCloseHistory = document.getElementById('btnCloseHistory');
  const btnClearHistory = document.getElementById('btnClearHistory');
  const historyPanel = document.getElementById('historyPanel');

  const modelSelect = document.getElementById('modelSelect');
  const modelUsedNote = document.getElementById('modelUsedNote');
  const contextToggle = document.getElementById('contextToggle'); // UI only

  /* =========================
     State
  ========================= */
  const HISTORY_KEY = 'reseeta_history_v1';
  const HISTORY_LIMIT = 20;
  const MODEL_KEY = 'reseeta_model_choice';

  let working = false;
  let lastUploadedId = null;
  let currentXHR = null;

  /* =========================
     Model choice memory
  ========================= */
  function getSavedModel(){ return localStorage.getItem(MODEL_KEY) || 'vit'; }
  function saveModel(v){ localStorage.setItem(MODEL_KEY, v); }

  /* =========================
     History helpers
  ========================= */
  function loadHistory() {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
    catch { return []; }
  }
  function saveHistory(items) {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, HISTORY_LIMIT)));
  }
  function addHistoryItem({id, name, dataUrl, status, resultText}) {
    const items = loadHistory();
    items.unshift({
      id, name, dataUrl,
      status: status || 'uploaded',
      resultText: resultText || null,
      ts: Date.now()
    });
    saveHistory(items);
  }
  function updateHistoryItem(id, patch) {
    const items = loadHistory();
    const i = items.findIndex(x => x.id === id);
    if (i !== -1) {
      items[i] = { ...items[i], ...patch };
      saveHistory(items);
    }
  }
  function formatDate(ts) { return new Date(ts).toLocaleString(); }
  function escapeHtml(s){ return String(s).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }
  function shorten(s, n){ return s && s.length>n ? s.slice(0, n-1)+'…' : (s || ''); }

  function renderHistory() {
    const items = loadHistory();
    const el = historyPanel.querySelector('.history-body');
    if (!items.length) { el.innerHTML = '<em>No history yet.</em>'; return; }
    el.innerHTML = items.map(it => `
      <div class="history-item" data-id="${it.id}">
        <img src="${it.dataUrl}" alt="${escapeHtml(it.name)}">
        <div>
          <div class="title">${escapeHtml(it.name)}</div>
          <div class="meta">${it.status === 'converted' ? 'Converted' : 'Uploaded'} • ${formatDate(it.ts)}</div>
          ${it.resultText ? `<div class="meta">Result: ${escapeHtml(shorten(it.resultText, 80))}</div>` : ''}
        </div>
      </div>
    `).join('');

    el.querySelectorAll('.history-item').forEach(node => {
      node.addEventListener('click', () => {
        const id = node.getAttribute('data-id');
        const item = loadHistory().find(x => x.id === id);
        if (!item) return;
        previewImage.src = item.dataUrl;
        previewImage.style.display = 'block';
        resultBox.textContent = item.resultText || '';
        // Keep heading reflecting last known model if any
      });
    });
  }

  function clearHistory(alsoResetUI = false){
    if (!confirm('Clear all local history on this browser?')) return;
    localStorage.removeItem(HISTORY_KEY);
    const body = historyPanel.querySelector('.history-body');
    if (body) body.innerHTML = '<em>No history yet.</em>';
    if (alsoResetUI) {
      fileInput.value = '';
      document.getElementById('fileName').textContent = 'filename.png';
      resetUploadingUI();
    }
  }

  /* =========================
     UI helpers
  ========================= */
  function showProgressOnly() {
    progressCard.hidden = false;
    progressCard.classList.remove('is-hidden');
    [...uploadInner.children].forEach(el => {
      if (el !== progressCard) el.classList.add('is-hidden');
    });
    previewImage.style.display = 'none';
    previewImage.classList.add('is-hidden');
  }

  function showPreviewOnly() {
    progressCard.hidden = true;
    progressCard.classList.add('is-hidden');
    [...uploadInner.children].forEach(el => {
      if (el !== previewImage) el.classList.add('is-hidden');
      else el.classList.remove('is-hidden');
    });
    previewImage.style.display = 'block';
  }

  function resetUploadingUI() {
    progressBar.style.width = '0%';
    progressPercent.textContent = '0%';
    progressStatus.textContent = 'Uploading…';
    progressCard.hidden = true;

    previewImage.style.display = 'none';
    previewImage.classList.add('is-hidden');
    [...uploadInner.children].forEach(el => {
      if (el !== progressCard) el.classList.remove('is-hidden');
    });

    convertLoading.hidden = true;

    // Reset heading + result box
    resultHeading.textContent = 'Result Here';
    resultHeading.classList.remove('is-hidden');
    resultBox.textContent = '';

    if (currentXHR) { try { currentXHR.abort(); } catch {} currentXHR = null; }

    startBtn.disabled = !fileInput.files?.length;
    working = false;

    if (modelUsedNote) modelUsedNote.textContent = '';
  }

  function enterUploadingUI() {
    showProgressOnly();
    // We keep the heading visible; result box is hidden during processing
    resultBox.classList.add('is-hidden');
    document.body.classList.add('recognize-busy');
  }

  function updateContextToggleAvailability() {
    const isVit = (modelSelect?.value === 'vit');         // only ViT-CRNN can use lexicon
    const wrapper = document.querySelector('.group.toggle .switch');

    if (isVit) {
      contextToggle.disabled = false;
      wrapper?.classList.remove('is-disabled');
      wrapper?.setAttribute('title', 'Contextual database is available for ViT-CRNN');
    } else {
      // moving away from ViT => force OFF and disable
      contextToggle.checked = false;
      contextToggle.disabled = true;
      wrapper?.classList.add('is-disabled');
      wrapper?.setAttribute('title', 'Contextual database is available only for ViT-CRNN');
    }
  }

  /* =========================
     Upload + recognize
     ========================= */
  function uploadAndRecognize() {
    const file = fileInput.files?.[0];
    if (!file) return;

    enterUploadingUI();
    progressStatus.textContent = 'Uploading…';

    const fd = new FormData();
    const modelVal = modelSelect ? modelSelect.value : 'vit';
    const useContext = (modelVal === 'vit' && contextToggle.checked) ? '1' : '0';

    fd.append('file', file, file.name);
    fd.append('model', modelVal);
    fd.append('use_context', useContext);

    const xhr = new XMLHttpRequest();
    currentXHR = xhr;
    xhr.open('POST', API_URL, true);
    xhr.responseType = 'json';
    xhr.setRequestHeader('X-CSRF-TOKEN', document.querySelector("meta[name='csrf-token']").getAttribute('content'));

    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const p = Math.max(0, Math.min(100, (e.loaded / e.total) * 100));
      progressBar.style.width = p + '%';
      progressPercent.textContent = Math.round(p) + '%';
    };

    xhr.upload.onload = () => {
      progressBar.style.width = '100%';
      progressPercent.textContent = '100%';
      progressStatus.textContent = 'Processing…';
      showPreviewOnly();
      convertLoading.hidden = false;
    };

    xhr.onreadystatechange = () => {
      if (xhr.readyState !== 4) return;

      convertLoading.hidden = true;
      document.body.classList.remove('recognize-busy');
      working = false;

      try {
        if (xhr.status >= 200 && xhr.status < 300) {
          const data = xhr.response || {};
          if (data.ok === false) throw new Error(data.detail || data.error || 'Model service failed');

          // From API
          const used       = (data.model_used || modelSelect.value || '').toString().trim();
          const isVit      = used.toLowerCase() === 'vit';
          const ctxFromAPI = (typeof data.context_enabled !== 'undefined') ? !!data.context_enabled : null;

          // Fallback to the UI toggle if API didn't say
          const ctxFallback = (isVit && contextToggle.checked);
          const ctxOn = (ctxFromAPI !== null) ? ctxFromAPI : ctxFallback;

          // Prefer backend flags:
          const applied =
            (typeof data.lexicon_applied !== 'undefined')
              ? !!data.lexicon_applied
              : (typeof data.lexicon_applied_strict !== 'undefined')
                ? !!data.lexicon_applied_strict
                : (!!data.lexicon_changed && ctxOn); // last-ditch fallback

          const changed  = !!data.lexicon_changed;
          const text     = (data.text ?? data.prediction ?? data.text_raw ?? '');

          // Clear old content
          resultBox.classList.remove('is-hidden');
          resultBox.textContent = '';
          const mainText = document.createElement('div');
          mainText.textContent = text || '(empty)';
          resultBox.appendChild(mainText);

          // Contextual DB line (what users care about)
          const ctxLine = document.createElement('div');
          ctxLine.style.marginTop = '6px';
          ctxLine.style.color = '#506A6D';
          ctxLine.style.fontSize = '.9rem';
          ctxLine.textContent = `Contextual DB applied: ${applied}`;
          if (changed && !applied) ctxLine.textContent += ' (changed first token but not a full DB match)';
          resultBox.appendChild(ctxLine);

          // Optional debug crumbs (helpful while tuning)
          if (data.lexicon_info) {
            const { reason, first_raw, first_fixed } = data.lexicon_info;
            const dbg = document.createElement('div');
            dbg.style.marginTop = '2px';
            dbg.style.color = '#7a8b8c';
            dbg.style.fontSize = '.8rem';
            let extra = [];
            if (typeof reason !== 'undefined') extra.push(`Reason: ${reason}`);
            if (first_raw)   extra.push(`raw="${first_raw}"`);
            if (first_fixed) extra.push(`fixed="${first_fixed}"`);
            if (extra.length) {
              dbg.textContent = extra.join(' • ');
              resultBox.appendChild(dbg);
            }
          }

          // Heading + note
resultHeading.textContent = used ? `Result using ${used.toUpperCase()}` : 'Result';

// NOTE: For the footer note we show the UI state ONLY (your requirement)
const uiToggleOn = (isVit && contextToggle.checked);
if (modelUsedNote) {
  const onoff = isVit ? (uiToggleOn ? 'Context ON' : 'Context OFF') : 'Context N/A';
  modelUsedNote.textContent = used ? `Last model: ${used.toUpperCase()} • ${onoff}` : '';
}

          if (lastUploadedId) {
            updateHistoryItem(lastUploadedId, { status: 'converted', resultText: text, ts: Date.now() });
          }

        } else {
          const err = xhr.response?.detail || xhr.response?.error || xhr.statusText || 'Upload failed';
          throw new Error(err);
        }
      } catch (e) {
        resultBox.classList.remove('is-hidden');
        resultBox.textContent = (e?.message || 'Unexpected error');
        resultHeading.textContent = 'Result';
        if (modelUsedNote) modelUsedNote.textContent = '';
        if (lastUploadedId) {
          updateHistoryItem(lastUploadedId, { status: 'converted', resultText: '(error)', ts: Date.now() });
        }
      } finally {
        currentXHR = null;
      }
    };

    xhr.onerror = () => {
      convertLoading.hidden = true;
      working = false;
      resultBox.classList.remove('is-hidden');
      resultBox.textContent = 'Network error';
      resultHeading.textContent = 'Result';
      if (modelUsedNote) modelUsedNote.textContent = '';
      currentXHR = null;
    };

    xhr.onabort = () => {
      convertLoading.hidden = true;
      working = false;
      resultBox.textContent = '';
      resultHeading.textContent = 'Result Here';
      if (modelUsedNote) modelUsedNote.textContent = '';
      currentXHR = null;
    };

    xhr.send(fd);
  }

  /* =========================
     Events
     ========================= */
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    startBtn.disabled = !file;
    if (!file) return;

    const r = new FileReader();
    r.onload = ev => {
      const dataUrl = ev.target.result;
      previewImage.src = dataUrl;
      previewImage.style.display = 'block';
      [...uploadInner.children].forEach(el => {
        if (el !== previewImage && el !== progressCard) el.classList.add('is-hidden');
      });

      const id = (crypto.randomUUID && crypto.randomUUID()) || String(Date.now());
      lastUploadedId = id;
      addHistoryItem({ id, name: file.name, dataUrl, status: 'uploaded' });
    };
    r.readAsDataURL(file);

    document.getElementById('fileName').textContent = file.name;
  });

  startBtn.addEventListener('click', () => {
    if (working) return;
    if (!fileInput.files || !fileInput.files[0]) return;
    working = true;
    uploadAndRecognize();
  });

  cancelBtn.addEventListener('click', () => {
    if (currentXHR) currentXHR.abort();
    resetUploadingUI();
  });

  btnDeleteUpload.addEventListener('click', (e) => {
    e.preventDefault();
    if (currentXHR) currentXHR.abort();
    fileInput.value = '';
    document.getElementById('fileName').textContent = 'filename.png';
    resetUploadingUI();
  });

  btnHistory.addEventListener('click', () => {
    const isHidden = historyPanel.hasAttribute('hidden');
    if (isHidden) {
      renderHistory();
      historyPanel.removeAttribute('hidden');
      btnHistory.setAttribute('aria-expanded', 'true');
    } else {
      historyPanel.setAttribute('hidden', '');
      btnHistory.setAttribute('aria-expanded', 'false');
    }
  });

  btnCloseHistory.addEventListener('click', () => {
    historyPanel.setAttribute('hidden', '');
    btnHistory.setAttribute('aria-expanded', 'false');
  });

  btnClearHistory?.addEventListener('click', () => clearHistory(false));

  function initUI(){
    resetUploadingUI();
    if (modelSelect) {
      modelSelect.value = getSavedModel();
      updateContextToggleAvailability();
      modelSelect.addEventListener('change', () => {
        saveModel(modelSelect.value);
        updateContextToggleAvailability();           
      });
    }
  }
  initUI();
  window.addEventListener('pageshow', initUI);
</script>
</body>
</html>
