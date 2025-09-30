<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ReSeeta – Convert</title>

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="{{ asset('css/convert.css') }}">
  <meta name="csrf-token" content="{{ csrf_token() }}">
  <style>
    #startConvert:disabled { opacity:.6; cursor:not-allowed; }
    /* optional helper */
    .is-hidden { display:none !important; }
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
          <!-- Delete uploaded image -->
          <button type="button"
                  id="btnDeleteUpload"
                  class="icon-btn icon-delete"
                  aria-label="Remove uploaded photo"
                  title="Remove uploaded photo">
            <img src="{{ asset('assets/delete.png') }}" alt="Delete" />
          </button>

          <input id="fileInput" type="file" accept="image/*" hidden>
          <div class="upload-inner">
            <!-- Default upload UI -->
            <div class="upload-icon" aria-hidden="true">🖼️</div>
            <div class="upload-title">Upload Photo</div>
            <p class="upload-note">
              Maximum file size: 10&nbsp;MB. Only clear, scanned medical prescriptions are accepted.
            </p>
            <img id="previewImage" alt="Image Preview" />

            <!-- Uploading progress (hidden until started) -->
            <div class="progress-card" id="progressCard" hidden>
              <div class="progress-title">Uploading</div>

              <div class="progress-row">
                <div class="file-icon" aria-hidden="true">🖼️</div>
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
          <!-- History button -->
          <button type="button"
                  id="btnHistory"
                  class="icon-btn icon-history"
                  aria-expanded="false"
                  aria-controls="historyPanel"
                  aria-label="View recognition history"
                  title="View history">
            <img src="{{ asset('assets/history.png') }}" alt="History" />
          </button>

          <span class="placeholder">Result Here</span>
          <div class="result-box" id="resultBox"></div>

          <!-- Converting loader (hidden until processing stage) -->
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

      <div class="actions">
        <button id="startConvert" type="button" disabled>Recognize Prescription</button>
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
  // Use your Laravel route that proxies to Python
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
  const resultPlaceholder = document.querySelector('.pane.result .placeholder');
  const resultBox = document.getElementById('resultBox');

  const btnDeleteUpload = document.getElementById('btnDeleteUpload');
  const btnHistory = document.getElementById('btnHistory');
  const btnCloseHistory = document.getElementById('btnCloseHistory');
  const btnClearHistory = document.getElementById('btnClearHistory');
  const historyPanel = document.getElementById('historyPanel');

  /* =========================
     State
  ========================= */
  const HISTORY_KEY = 'reseeta_history_v1';
  const HISTORY_LIMIT = 20;
  let uploadTimer = null;       // (not used with XHR progress, kept for compatibility)
  let working = false;          // prevents double-submit
  let lastUploadedId = null;    // history link
  let currentXHR = null;        // in-flight request (for cancel)

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
      status: status || 'uploaded', // 'uploaded' | 'converted'
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
  function shorten(s, n){ return s.length>n ? s.slice(0, n-1)+'…' : s; }

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
        resultPlaceholder.classList.toggle('is-hidden', !!item.resultText);
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
    // keep preview visible only after selection (not during upload here)
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

    // Keep spinner hidden until processing stage
    convertLoading.hidden = true;

    resultPlaceholder?.classList.remove('is-hidden');
    resultBox?.classList.remove('is-hidden');
    resultBox.textContent = '';

    document.body.classList.remove('recognize-busy');
    if (uploadTimer) { clearInterval(uploadTimer); uploadTimer = null; }
    working = false;

    if (currentXHR) { try { currentXHR.abort(); } catch {} currentXHR = null; }

    startBtn.disabled = !fileInput.files?.length;
  }

  function enterUploadingUI() {
    showProgressOnly();
    // spinner appears only during "processing", not upload
    resultPlaceholder?.classList.add('is-hidden');
    resultBox?.classList.add('is-hidden');
    document.body.classList.add('recognize-busy');
  }

  /* =========================
     Upload + recognize
  ========================= */
  async function uploadAndRecognize() {
    const file = fileInput.files?.[0];
    if (!file) return;

    enterUploadingUI();
    progressStatus.textContent = 'Uploading…';

    const fd = new FormData();
    // IMPORTANT: field name must be 'file' to match OcrController
    fd.append('file', file, file.name);

    currentXHR = new XMLHttpRequest();
    currentXHR.open('POST', API_URL, true);
    currentXHR.responseType = 'json';
    // CSRF header for Laravel
    currentXHR.setRequestHeader('X-CSRF-TOKEN', document.querySelector("meta[name='csrf-token']").getAttribute('content'));

    // upload progress
    currentXHR.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const p = Math.max(0, Math.min(100, (e.loaded / e.total) * 100));
      progressBar.style.width = p + '%';
      progressPercent.textContent = Math.round(p) + '%';
    };

    // after upload completes, switch to processing spinner and keep preview visible
    currentXHR.upload.onload = () => {
      progressBar.style.width = '100%';
      progressPercent.textContent = '100%';
      progressStatus.textContent = 'Processing…';
      showPreviewOnly();          // preview stays visible during processing
      convertLoading.hidden = false; // show spinner now
    };

    currentXHR.onreadystatechange = () => {
      if (currentXHR.readyState !== 4) return;

      convertLoading.hidden = true;
      document.body.classList.remove('recognize-busy');
      working = false;

      try {
        if (currentXHR.status >= 200 && currentXHR.status < 300) {
          const data = currentXHR.response || {};
          if (data.ok === false) throw new Error(data.detail || data.error || 'Model service failed');

          const text = data.text || '(empty)';
          resultPlaceholder?.classList.add('is-hidden');
          resultBox.classList.remove('is-hidden');
          resultBox.textContent = text;

          if (lastUploadedId) {
            updateHistoryItem(lastUploadedId, { status: 'converted', resultText: text, ts: Date.now() });
          }
        } else {
          const err = currentXHR.response?.detail || currentXHR.response?.error || currentXHR.statusText || 'Upload failed';
          throw new Error(err);
        }
      } catch (e) {
        resultPlaceholder?.classList.add('is-hidden');
        resultBox.classList.remove('is-hidden');
        resultBox.textContent = '❌ ' + (e?.message || 'Unexpected error');
        if (lastUploadedId) {
          updateHistoryItem(lastUploadedId, { status: 'converted', resultText: '(error)', ts: Date.now() });
        }
      } finally {
        currentXHR = null;
      }
    };

    currentXHR.onerror = () => {
      convertLoading.hidden = true;
      working = false;
      resultBox.textContent = '❌ Network error';
      currentXHR = null;
    };

    currentXHR.onabort = () => {
      convertLoading.hidden = true;
      working = false;
      resultBox.textContent = '';
      currentXHR = null;
    };

    currentXHR.send(fd);
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
      // Show preview immediately; hide the upload UI
      previewImage.src = dataUrl;
      previewImage.style.display = 'block';
      [...uploadInner.children].forEach(el => {
        if (el !== previewImage && el !== progressCard) el.classList.add('is-hidden');
      });

      // history
      lastUploadedId = (crypto.randomUUID && crypto.randomUUID()) || String(Date.now());
      addHistoryItem({
        id: lastUploadedId,
        name: file.name,
        dataUrl,
        status: 'uploaded'
      });
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

  // Init on load & BFCache restore
  resetUploadingUI();
  window.addEventListener('pageshow', resetUploadingUI);
</script>
</body>
</html>
