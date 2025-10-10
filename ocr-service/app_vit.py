# app_vit.py
import os, csv, re, math, io
from pathlib import Path
from typing import List

import numpy as np
import torch
import cv2
import importlib.util
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

# -----------------------------
# Model / tokenization constants
# -----------------------------
IMG_H, IMG_W = 128, 1024
BLANK_ID = 0

import string
charset_base = string.printable[:95]
VOCAB_SIZE = len(charset_base) + 1

# -----------------------------
# Tokenizer (CTC)
# -----------------------------
class SimpleTokenizer:
    def __init__(self, charset: str, blank_id: int = 0):
        self.i2c = {i + 1: c for i, c in enumerate(list(charset))}
        self.blank = blank_id

    def decode_ids(self, ids: List[int]) -> str:
        return "".join(self.i2c.get(int(i), "") for i in ids if int(i) != self.blank)

tokenizer = SimpleTokenizer(charset_base, blank_id=BLANK_ID)

# -----------------------------
# Import model module from path
# -----------------------------
def import_from_path(py_path: str, module_name: str = "reseeta_model"):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# -----------------------------
# CTC greedy decode
# -----------------------------
def greedy_ids(logp_T_C: torch.Tensor, blank_id: int = 0) -> List[int]:
    ids = logp_T_C.argmax(dim=-1).cpu().numpy()
    out, prev = [], None
    for k in ids:
        k = int(k)
        if k != blank_id and k != prev:
            out.append(k)
        prev = k
    return out

# --- Lexicon (first-word correction) ---
def _choose_lexicon_path() -> Path:
    env = os.getenv("DRUG_LEXICON_CSV")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p

    here = Path(__file__).resolve().parent
    candidates = [
        here / "cleaned_drug_names.csv",                      # same folder as app_vit.py
        here / "ocr-service" / "cleaned_drug_names.csv",      # subfolder next to app
        here.parent / "ocr-service" / "cleaned_drug_names.csv",
    ]
    for c in candidates:
        if c.is_file():
            return c

    return candidates[0]

LEXICON_PATH = _choose_lexicon_path()
LEXICON_CSV = str(LEXICON_PATH)

def _load_lexicon(csv_path: str):
    names = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if not r.fieldnames or "drug_name" not in r.fieldnames:
                raise ValueError(f"CSV at {csv_path} must contain a 'drug_name' header")
            for row in r:
                name = (row["drug_name"] or "").strip()
                if name:
                    names.append(name)
    except FileNotFoundError:
        print(f"⚠️ Lexicon CSV not found at: {csv_path}. First-word correction will be skipped.")
    return names

_DRUG_NAMES = _load_lexicon(LEXICON_CSV)
_LEX_FIRST_TOKENS = [(n.split()[0], n) for n in _DRUG_NAMES]
_LEX_FIRST_TOKENS_LOWER = [(t.lower(), full) for t, full in _LEX_FIRST_TOKENS]
print(f"[lexicon] {len(_DRUG_NAMES)} entries loaded from: {LEXICON_CSV}")

# Sets for quick checks
_LEX_SET_LOWER = {n.strip().lower() for n in _DRUG_NAMES}
_LEX_FIRST_TOKEN_SET = {t.lower() for (t, _full) in _LEX_FIRST_TOKENS}

def _levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        ai = a[i - 1]
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ai != b[j - 1]))
            prev = cur
    return dp[n]

_WORD_RE = re.compile(r"[A-Za-z0-9\-\.\+]+")

def _first_word(text: str) -> str:
    if not text: return ""
    m = _WORD_RE.search(text)
    return m.group(0) if m else ""

def _replace_first_word(text: str, new_first: str) -> str:
    m = _WORD_RE.search(text)
    if not m:
        return text if text else new_first
    s, e = m.span()
    return text[:s] + new_first + text[e:]

def _nearest_lex_first_token(word: str):
    if not word or not _LEX_FIRST_TOKENS_LOWER:
        return None, None, 1.0
    w = word.lower()
    best = (None, None, 1.0)  # token, full_name, norm_dist
    for tok_lower, full_name in _LEX_FIRST_TOKENS_LOWER:
        d = _levenshtein(w, tok_lower)
        nd = d / max(1, max(len(w), len(tok_lower)))
        if nd < best[2]:
            orig_token = full_name.split()[0]
            best = (orig_token, full_name, nd)
    return best

def maybe_fix_first_word(pred: str):
    """Returns (fixed_pred, changed_bool, info_dict)."""
    if not _DRUG_NAMES:
        return pred, False, {"applied": False, "reason": "no-lexicon"}

    first = _first_word(pred)
    if not first:
        return pred, False, {"applied": False, "reason": "no-first-word", "first": ""}

    first_lower = first.lower()
    for tok_lower, full_name in _LEX_FIRST_TOKENS_LOWER:
        if first_lower == tok_lower:
            return pred, False, {"applied": True, "reason": "exact-match", "first": first}

    best_tok, best_full, nd = _nearest_lex_first_token(first)
    same_initial = (first_lower[:1] == (best_tok or "").lower()[:1])
    max_nd = 0.34 if len(first) >= 6 else (0.25 if len(first) >= 4 else 0.20)

    if best_tok and same_initial and nd <= max_nd:
        fixed = _replace_first_word(pred, best_tok)
        return fixed, True, {
            "applied": True,
            "reason": f"nearest(nd={nd:.3f})",
            "first": first,
            "candidate": best_tok,
            "full": best_full,
            "nd": nd
        }

    return pred, False, {
        "applied": True,
        "reason": f"no-good-candidate(nd={nd:.3f})",
        "first": first,
        "candidate": best_tok,
        "nd": nd
    }

# ==========================================================
#                 PREPROCESSING
# ==========================================================
DENOISE_STRENGTH = int(os.getenv("PP_DENOISE_STRENGTH", 7))
DENOISE_TEMPLATE = int(os.getenv("PP_DENOISE_TEMPLATE", 7))
DENOISE_SEARCH   = int(os.getenv("PP_DENOISE_SEARCH",   21))

GAUSS_BLUR_KSIZE = int(os.getenv("PP_GAUSS_KSIZE", 3))
GAUSS_BLUR_SIGMA = int(os.getenv("PP_GAUSS_SIGMA", 0))

CANNY_T1       = int(os.getenv("PP_CANNY_T1", 50))
CANNY_T2       = int(os.getenv("PP_CANNY_T2", 150))
CANNY_APERTURE = int(os.getenv("PP_CANNY_APERTURE", 3))
USE_L2GRAD     = os.getenv("PP_CANNY_L2", "1") not in ("0", "false", "False")

OUTPUT_MODE    = os.getenv("PP_OUTPUT_MODE", "edges_only").strip().lower()
INVERT_OUTPUT  = os.getenv("PP_INVERT", "1") not in ("0", "false", "False")

def _normalize_0_255(img_gray_u8: np.ndarray) -> np.ndarray:
    img = img_gray_u8.astype(np.float32)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi - lo < 1e-3:
        return np.uint8(np.clip(img, 0, 255))
    img = (img - lo) * (255.0 / (hi - lo))
    return np.uint8(np.clip(img, 0, 255))

def preprocess_and_fuse(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    den = cv2.fastNlMeansDenoising(
        gray, None,
        h=DENOISE_STRENGTH,
        templateWindowSize=DENOISE_TEMPLATE,
        searchWindowSize=DENOISE_SEARCH
    )
    norm = _normalize_0_255(den)

    blur = norm
    if GAUSS_BLUR_KSIZE and GAUSS_BLUR_KSIZE >= 3 and GAUSS_BLUR_KSIZE % 2 == 1:
        blur = cv2.GaussianBlur(norm, (GAUSS_BLUR_KSIZE, GAUSS_BLUR_KSIZE), GAUSS_BLUR_SIGMA)

    edges = cv2.Canny(blur, CANNY_T1, CANNY_T2, apertureSize=CANNY_APERTURE, L2gradient=USE_L2GRAD)

    if OUTPUT_MODE == "edges_only":
        final = edges
        if INVERT_OUTPUT:
            final = cv2.bitwise_not(final)
    elif OUTPUT_MODE == "overlay":
        if INVERT_OUTPUT:
            final = np.full_like(norm, 255, dtype=np.uint8)
            final[edges > 0] = 0
        else:
            final = norm.copy()
            final[edges > 0] = 255
    elif OUTPUT_MODE == "norm_only":
        final = norm
        if INVERT_OUTPUT:
            final = cv2.bitwise_not(final)
    else:
        raise ValueError(f"Unknown OUTPUT_MODE: {OUTPUT_MODE}")

    return final  # uint8

# -----------------------------
# Fit to 1024x128 canvas (keep aspect)
# -----------------------------
def fit_to_canvas_1024x128_u8(img_u8: np.ndarray) -> np.ndarray:
    h0, w0 = img_u8.shape
    if (h0, w0) == (IMG_H, IMG_W):
        return img_u8
    scale = min(IMG_W / w0, IMG_H / h0)
    nw = max(1, int(round(w0 * scale)))
    nh = max(1, int(round(h0 * scale)))
    resized = cv2.resize(img_u8, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((IMG_H, IMG_W), 255, np.uint8)
    y0 = (IMG_H - nh) // 2
    x0 = (IMG_W - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

# -----------------------------
# Load model once (warm)
# -----------------------------
MODEL_PY = os.environ.get("MODEL_PY", "reseeta_model.py")
WEIGHTS  = os.environ.get("WEIGHTS",   "ViT_CRNN_weights.pth")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m = import_from_path(MODEL_PY)
cfg = m.ViTCRNNConfig(in_ch=1, num_classes=VOCAB_SIZE, patch_w=1, norm_first=True)
vit_model = m.ViTCRNN(cfg)  # <-- renamed to avoid shadowing
with torch.no_grad():
    _ = vit_model.forward(torch.zeros(1, 1, IMG_H, IMG_W))
sd = torch.load(WEIGHTS, map_location="cpu")
state = sd.get("model", sd) if isinstance(sd, dict) else sd
vit_model.load_state_dict(state, strict=False)
vit_model.to(DEVICE).eval()

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "lexicon": {
            "count": len(_DRUG_NAMES),
            "path": str(LEXICON_CSV),
        },
        "preproc": {
            "mode": OUTPUT_MODE,
            "invert": bool(INVERT_OUTPUT),
            "denoise": [DENOISE_STRENGTH, DENOISE_TEMPLATE, DENOISE_SEARCH],
            "gauss": [GAUSS_BLUR_KSIZE, GAUSS_BLUR_SIGMA],
            "canny": [CANNY_T1, CANNY_T2, CANNY_APERTURE, bool(USE_L2GRAD)],
            "canvas": [IMG_H, IMG_W],
        }
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_choice: str = Form("vit", alias="model"),   # <-- accept field named "model"
    use_context: str = Form("0"),
):
    
    # DEBUG: log what we received
    print(f"[predict] model={model_choice!r} use_context_raw={use_context!r}")

    # 1) Read & decode
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    # 2) Preprocess
    try:
        fused_u8 = preprocess_and_fuse(bgr)
    except Exception as e:
        return JSONResponse({"error": f"Preprocessing failed: {e}"}, status_code=400)

    # 3) Canvas
    canvas_u8 = fit_to_canvas_1024x128_u8(fused_u8)

    # 4) Tensor
    canvas = canvas_u8.astype(np.float32) / 255.0
    x = torch.from_numpy(canvas[None, None, ...]).float().to(DEVICE)

    # 5) Predict (use vit_model, not the form field!)
    with torch.inference_mode():
        logp = vit_model.log_probs(x)      # (T,B,C)
        logp_single = logp[:, 0, :]
        ids = greedy_ids(logp_single, blank_id=BLANK_ID)
        text_raw = tokenizer.decode_ids(ids)

    # 6) Optional first-word lexicon correction
    context_enabled = (use_context in ("1", "true", "True"))
    if context_enabled:
        text_fixed, changed, info = maybe_fix_first_word(text_raw)
    else:
        text_fixed, changed, info = text_raw, False, {"applied": False, "reason": "context-off"}

    # Applied signal for UI
    first_raw   = _first_word(text_raw).strip()
    first_fixed = _first_word(text_fixed).strip()
    lexicon_applied = (
        bool(context_enabled) and
        bool(changed) and
        (first_fixed.lower() in _LEX_FIRST_TOKEN_SET)
    )
    lexicon_applied_strict = bool(text_fixed.strip().lower() in _LEX_SET_LOWER)

    return {
        "ok": True,
        "model_used": "vit",
        "text_raw": text_raw,
        "text": text_fixed,

        "context_enabled": bool(context_enabled),
        "use_context_raw": use_context,
        "lexicon_changed": bool(changed),
        "lexicon_applied": bool(lexicon_applied),
        "lexicon_applied_strict": bool(lexicon_applied_strict),

        "lexicon_info": {
            **(info if isinstance(info, dict) else {"reason": str(info)}),
            "first_raw": first_raw,
            "first_fixed": first_fixed,
            "first_fixed_in_lex": (first_fixed.lower() in _LEX_FIRST_TOKEN_SET),
        },
        "lexicon_count": len(_DRUG_NAMES),
        "lexicon_path": str(LEXICON_CSV),

        "shape": [int(s) for s in canvas_u8.shape],
        "preproc_mode": OUTPUT_MODE,
        "inverted": bool(INVERT_OUTPUT),
    }

if __name__ == "__main__":
    # If you ever run: python app_vit.py
    uvicorn.run("app_vit:app", host="0.0.0.0", port=8001, reload=False)
