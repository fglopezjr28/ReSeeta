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
    /* (kept your in-file CSS tweaks exactly as before; omitted here to keep snippet compact in message) */
    /* Place the same style block you used previously here if you want to keep it inline. */
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
      <a href="{{ url('/vit-crnn-results') }}" class="{{ Request::is('vit-crnn-results') ? 'active' : '' }}">ViT-CRNN Results</a>
      <a href="{{ url('/crnn-results') }}" class="{{ Request::is('crnn-results') ? 'active' : '' }}">CRNN Results</a>
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

          <span class="placeholder" id="resultHeading">Result Here</span>

          <div class="result-box" id="resultBox" aria-live="polite"></div>

          <!-- Small, privacy-friendly debug pill (no raw JSON) -->
          <div class="result-debug" id="resultDebug" aria-live="polite"></div>

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

        <span id="modelUsedNote" class="model-note" aria-live="polite"></span>
      </div>
    </section>
  </main>

  <footer>
    <p>© {{ date('Y') }} ReSeeta. All Rights Reserved.</p>
  </footer>

<script>
/* =========================
   Config & helpers
   ========================= */
const API_URL = "{{ route('ocr.predict') }}" || "/predict_both";

function escapeHtml(s){
  return String(s || '').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

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
const resultDebug = document.getElementById('resultDebug');

const btnDeleteUpload = document.getElementById('btnDeleteUpload');
const btnHistory = document.getElementById('btnHistory');
const btnCloseHistory = document.getElementById('btnCloseHistory');
const btnClearHistory = document.getElementById('btnClearHistory');
const historyPanel = document.getElementById('historyPanel');

const modelSelect = document.getElementById('modelSelect');
const modelUsedNote = document.getElementById('modelUsedNote');
const contextToggle = document.getElementById('contextToggle');

/* =========================
   State + small utils
   ========================= */
const HISTORY_KEY = 'reseeta_history_v1';
const HISTORY_LIMIT = 20;
const MODEL_KEY = 'reseeta_model_choice';

let working = false;
let lastUploadedId = null;
let currentXHR = null;

function getSavedModel(){ return localStorage.getItem(MODEL_KEY) || 'vit'; }
function saveModel(v){ localStorage.setItem(MODEL_KEY, v); }

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
    const filenameEl = document.getElementById('fileName');
    if (filenameEl) filenameEl.textContent = 'filename.png';
    resetUploadingUI();
  }
}

/* =========================
   UI helpers
   ========================= */
function showProgressOnly() {
  if (!progressCard) return;
  progressCard.hidden = false;
  progressCard.classList.remove('is-hidden');
  [...uploadInner.children].forEach(el => {
    if (el !== progressCard) el.classList.add('is-hidden');
  });
  if (previewImage) {
    previewImage.style.display = 'none';
    previewImage.classList.add('is-hidden');
  }
}

function showPreviewOnly() {
  if (!progressCard) return;
  progressCard.hidden = true;
  progressCard.classList.add('is-hidden');
  [...uploadInner.children].forEach(el => {
    if (el !== previewImage) el.classList.add('is-hidden');
    else el.classList.remove('is-hidden');
  });
  if (previewImage) previewImage.style.display = 'block';
}

function resetUploadingUI() {
  if (progressBar) progressBar.style.width = '0%';
  if (progressPercent) progressPercent.textContent = '0%';
  if (progressStatus) progressStatus.textContent = 'Uploading…';
  if (progressCard) progressCard.hidden = true;

  if (previewImage) {
    previewImage.style.display = 'none';
    previewImage.classList.add('is-hidden');
  }
  [...uploadInner.children].forEach(el => {
    if (el !== progressCard) el.classList.remove('is-hidden');
  });

  if (convertLoading) convertLoading.hidden = true;

  if (resultHeading) resultHeading.textContent = 'Result Here';
  if (resultHeading) resultHeading.classList.remove('is-hidden');
  if (resultBox) resultBox.textContent = '';
  if (resultDebug) resultDebug.innerHTML = '';

  if (currentXHR) { try { currentXHR.abort(); } catch {} currentXHR = null; }

  startBtn.disabled = !fileInput.files?.length;
  working = false;

  if (modelUsedNote) modelUsedNote.textContent = '';
}

function enterUploadingUI() {
  showProgressOnly();
  if (resultBox) resultBox.classList.add('is-hidden');
  document.body.classList.add('recognize-busy');
}

function updateContextToggleAvailability() {
  const isVit = (modelSelect?.value === 'vit');
  const wrapper = document.querySelector('.group.toggle .switch');

  if (isVit) {
    contextToggle.disabled = false;
    wrapper?.classList.remove('is-disabled');
    wrapper?.setAttribute('title', 'Contextual database is available for ViT-CRNN');
  } else {
    contextToggle.checked = false;
    contextToggle.disabled = true;
    wrapper?.classList.add('is-disabled');
    wrapper?.setAttribute('title', 'Contextual database is available only for ViT-CRNN');
  }
}

/* =========================
   Sanitized renderer (NO raw JSON in DOM)
   ========================= */
function renderSanitizedResult(body, httpStatus) {
  // Prefer remote.assembled -> local.text -> direct text
  let displayText = null;
  if (body && body.remote && typeof body.remote === 'object') {
    if (typeof body.remote.assembled !== 'undefined') displayText = body.remote.assembled;
    else if (typeof body.remote.text !== 'undefined') displayText = body.remote.text;
  }
  if (!displayText && body && typeof body.assembled !== 'undefined') displayText = body.assembled;
  if (!displayText && body && body.local) displayText = body.local.text || body.local.text_raw || null;
  if (!displayText && body && (body.text || body.text_raw)) displayText = body.text || body.text_raw;

  if (displayText === null || typeof displayText === 'undefined' || String(displayText).trim() === '') {
    displayText = '(no extracted text)';
  } else {
    displayText = String(displayText).trim();
  }

  if (resultBox) {
    resultBox.classList.remove('is-hidden');
    resultBox.textContent = displayText;
  }

  let modelUsed = (body && (body.model_used || (body.local && body.local.model_used) || (body.remote && body.remote.model_used))) || '';
  modelUsed = String(modelUsed).toUpperCase();

  let remoteOk = false;
  if (body && body.remote && typeof body.remote === 'object') {
    if ((typeof body.remote.assembled !== 'undefined' && String(body.remote.assembled).trim() !== "") ||
        (body.remote.full_result && (body.remote.full_result.status === "DONE" || (body.remote.full_result.status_message && /done|success/i.test(body.remote.full_result.status_message))))) {
      remoteOk = true;
    }
  }

  let medConf = null;
  try {
    if (body && body.remote && body.remote.full_result && body.remote.full_result.Line_fields && body.remote.full_result.Line_fields["Medication name"]) {
      const meds = body.remote.full_result.Line_fields["Medication name"];
      if (Array.isArray(meds) && meds.length && meds[0].confidence_score !== undefined) {
        medConf = Number(meds[0].confidence_score);
      }
    }
  } catch (e) { medConf = null; }

  let badgeText = modelUsed ? modelUsed : ' ';
  let details = remoteOk ? ' ' : ' ';
  if (medConf !== null && !Number.isNaN(medConf)) {
    details += ` `;
  }

  if (resultDebug) {
    const pill = `<span class="badge">${escapeHtml(badgeText)}</span>
      <div class="details">${escapeHtml(details)}</div>`;
    resultDebug.innerHTML = pill;
  }

  // Developer console still gets the full object for debugging (not in DOM)
  console.debug("OCR sanitized:", httpStatus, "displayText:", displayText, "rawBody:", body);
}

/* =========================
   Upload & recognize (XHR -> then fetch saved JSON)
   ========================= */
function uploadAndRecognize() {
  const file = fileInput.files?.[0];
  if (!file) return;

  enterUploadingUI();
  if (progressStatus) progressStatus.textContent = 'Uploading…';

  const fd = new FormData();
  const modelVal = modelSelect ? modelSelect.value : 'vit';
  const useContext = (modelVal === 'vit' && contextToggle && contextToggle.checked) ? '1' : '0';

  fd.append('file', file, file.name);
  fd.append('model', modelVal);
  fd.append('use_context', useContext);

  const xhr = new XMLHttpRequest();
  currentXHR = xhr;
  xhr.open('POST', API_URL, true);
  xhr.responseType = 'json';

  try {
    const tokenMeta = document.querySelector("meta[name='csrf-token']");
    if (tokenMeta && API_URL.startsWith(window.location.origin)) {
      xhr.setRequestHeader('X-CSRF-TOKEN', tokenMeta.getAttribute('content'));
    }
  } catch (e) {}

  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const p = Math.max(0, Math.min(100, (e.loaded / e.total) * 100));
    if (progressBar) progressBar.style.width = p + '%';
    if (progressPercent) progressPercent.textContent = Math.round(p) + '%';
  };

  xhr.upload.onload = () => {
    if (progressBar) progressBar.style.width = '100%';
    if (progressPercent) progressPercent.textContent = '100%';
    if (progressStatus) progressStatus.textContent = 'Processing…';
    showPreviewOnly();
    if (convertLoading) convertLoading.hidden = false;
  };

  xhr.onreadystatechange = () => {
    if (xhr.readyState !== 4) return;

    if (convertLoading) convertLoading.hidden = true;
    document.body.classList.remove('recognize-busy');
    working = false;

    // Non-2xx
    if (xhr.status < 200 || xhr.status >= 300) {
      const raw = xhr.response || (xhr.responseText ? xhr.responseText : null);
      const errMessage = (raw && (raw.error || raw.detail || raw.message)) || xhr.statusText || 'Upload failed';
      if (resultBox) {
        resultBox.classList.remove('is-hidden');
        resultBox.textContent = `Error: ${errMessage}`;
      }
      if (resultDebug) {
        try {
          const bodyWrapper = (raw && typeof raw === 'object') ? raw : { raw_text: raw };
          resultDebug.innerHTML = `<span class="badge">HTTP ${xhr.status}</span>
            <div class="details">${escapeHtml(String(errMessage))}</div>`;
        } catch (e) { resultDebug.textContent = String(e); }
      }
      currentXHR = null;
      return;
    }

    // Success: fetch the JSON file Laravel wrote (cache-busted)
    const jsonUrl = '/ocr_result.json?ts=' + Date.now();
    fetch(jsonUrl, { cache: "no-store" })
      .then(r => {
        if (!r.ok) throw new Error("Could not fetch result JSON: " + r.status);
        return r.json();
      })
      .then((body) => {
        // sanitize & render only summary
        renderSanitizedResult(body, xhr.status);
        if (lastUploadedId) updateHistoryItem(lastUploadedId, { status: 'converted', resultText: (body && body.remote && body.remote.assembled) || (body && body.local && body.local.text) || '(no extracted text)', ts: Date.now() });
      })
      .catch(err => {
        console.error("Failed to fetch/read ocr_result.json:", err);
        if (resultBox) {
          resultBox.classList.remove('is-hidden');
          resultBox.textContent = "(error reading result file)";
        }
        if (resultDebug) {
          resultDebug.innerHTML = `<span class="badge">File error</span><div class="details">${escapeHtml(String(err))}</div>`;
        }
      })
      .finally(() => {
        currentXHR = null;
      });
  };

  xhr.onerror = () => {
    if (convertLoading) convertLoading.hidden = true;
    working = false;
    if (resultBox) {
      resultBox.classList.remove('is-hidden');
      resultBox.textContent = 'Network error';
    }
    if (resultDebug) resultDebug.innerHTML = '';
    if (resultHeading) resultHeading.textContent = 'Result';
    if (modelUsedNote) modelUsedNote.textContent = '';
    currentXHR = null;
  };

  xhr.onabort = () => {
    if (convertLoading) convertLoading.hidden = true;
    working = false;
    if (resultBox) resultBox.textContent = '';
    if (resultDebug) resultDebug.innerHTML = '';
    if (resultHeading) resultHeading.textContent = 'Result Here';
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

  const filenameEl = document.getElementById('fileName');
  if (filenameEl) filenameEl.textContent = file.name;
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
  const filenameEl = document.getElementById('fileName');
  if (filenameEl) filenameEl.textContent = 'filename.png';
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
