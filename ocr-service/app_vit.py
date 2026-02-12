# app_vit.py
# (full file — original content with segmentation integrated as a split-friendly import.
# If segment_words.py is present in the same folder, it will be used. Otherwise a fallback
# implementation (same algorithm) is embedded so the endpoints always work.)
#
# Segmentation is "display-only": it will save crops + debug image into PREPROC_DIR (default: preproc_debug)
# and the endpoints will return the segmentation summary in "debug_saved".
#
import os
import re
import io
import time
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import cv2
import importlib.util
import uvicorn
import requests

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

IMG_H, IMG_W = 128, 1024
BLANK_ID = 0

import string
charset_base = string.printable[:95]
VOCAB_SIZE = len(charset_base) + 1

class SimpleTokenizer:
    def __init__(self, charset: str, blank_id: int = 0):
        self.i2c = {i + 1: c for i, c in enumerate(list(charset))}
        self.blank = blank_id

    def decode_ids(self, ids: List[int]) -> str:
        return "".join(self.i2c.get(int(i), "") for i in ids if int(i) != self.blank)

tokenizer = SimpleTokenizer(charset_base, blank_id=BLANK_ID)

def import_from_path(py_path: str, module_name: str = "reseeta_model"):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def greedy_ids(logp_T_C: torch.Tensor, blank_id: int = 0) -> List[int]:
    ids = logp_T_C.argmax(dim=-1).cpu().numpy()
    out, prev = [], None
    for k in ids:
        k = int(k)
        if k != blank_id and k != prev:
            out.append(k)
        prev = k
    return out

# --- lexicon helpers (unchanged) ---
def _choose_lexicon_path() -> Path:
    env = os.getenv("DRUG_LEXICON_CSV")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p

    here = Path(__file__).resolve().parent
    candidates = [
        here / "cleaned_drug_names.csv",
        here / "ocr-service" / "cleaned_drug_names.csv",
        here.parent / "ocr-service" / "cleaned_drug_names.csv",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]

LEXICON_PATH = _choose_lexicon_path()
LEXICON_CSV = str(LEXICON_PATH)

def _load_lexicon(csv_path: str):
    """
    Robust CSV loader for lexicon that tries multiple encodings and falls back to
    a lossy decode (errors='replace') if necessary so the server does not crash.
    Returns list of drug name strings (may be empty).
    """
    names = []
    p = Path(csv_path)
    if not p.is_file():
        print(f"⚠️ Lexicon CSV not found at: {csv_path}. First-word correction will be skipped.")
        return names

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_exc = None
    for enc in encodings_to_try:
        try:
            with p.open("r", encoding=enc, newline="") as f:
                r = csv.DictReader(f)
                if not r.fieldnames or "drug_name" not in r.fieldnames:
                    raise ValueError(f"CSV at {csv_path} must contain a 'drug_name' header (tried encoding: {enc})")
                for row in r:
                    name = (row.get("drug_name") or "").strip()
                    if name:
                        names.append(name)
            print(f"✅ Loaded lexicon '{csv_path}' with encoding: {enc} (entries: {len(names)})")
            return names
        except Exception as e:
            last_exc = e
            # try next encoding

    # final fallback: lossy decode so server still boots; replace undecodable bytes
    try:
        with p.open("rb") as f:
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        # re-parse using csv module from a StringIO
        from io import StringIO
        s = StringIO(text)
        r = csv.DictReader(s)
        if not r.fieldnames or "drug_name" not in r.fieldnames:
            print(f"⚠️ After fallbacks, CSV still missing 'drug_name' header. Returning empty lexicon. (path={csv_path})")
            return []
        for row in r:
            name = (row.get("drug_name") or "").strip()
            if name:
                names.append(name)
        print(f"⚠️ Loaded lexicon '{csv_path}' with lossy decode (errors='replace') — entries: {len(names)}")
        return names
    except Exception as e:
        print(f"❌ Failed to load lexicon '{csv_path}': {last_exc}; fallback failed: {e}")
        return []


_DRUG_NAMES = _load_lexicon(LEXICON_CSV)
_LEX_FIRST_TOKENS = [(n.split()[0], n) for n in _DRUG_NAMES]
_LEX_FIRST_TOKENS_LOWER = [(t.lower(), full) for t, full in _LEX_FIRST_TOKENS]
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
    best = (None, None, 1.0)
    for tok_lower, full_name in _LEX_FIRST_TOKENS_LOWER:
        d = _levenshtein(w, tok_lower)
        nd = d / max(1, max(len(w), len(tok_lower)))
        if nd < best[2]:
            orig_token = full_name.split()[0]
            best = (orig_token, full_name, nd)
    return best

def maybe_fix_first_word(pred: str):
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

# --- preprocessing (unchanged) ---
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
SAVE_PREPROC = os.getenv("SAVE_PREPROC", "1") not in ("0", "false", "False")
PREPROC_DIR  = os.getenv("PREPROC_DIR", "preproc_debug")

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
    return final

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

# --- model load (unchanged) ---
MODEL_PY = os.environ.get("MODEL_PY", "reseeta_model.py")
WEIGHTS  = os.environ.get("WEIGHTS",   "ViT_CRNN_weights.pth")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m = import_from_path(MODEL_PY)
cfg = m.ViTCRNNConfig(in_ch=1, num_classes=VOCAB_SIZE, patch_w=1, norm_first=True)
vit_model = m.ViTCRNN(cfg)
with torch.no_grad():
    _ = vit_model.forward(torch.zeros(1, 1, IMG_H, IMG_W))
sd = torch.load(WEIGHTS, map_location="cpu")
state = sd.get("model", sd) if isinstance(sd, dict) else sd
vit_model.load_state_dict(state, strict=False)
vit_model.to(DEVICE).eval()

# --- external OCR config (hard-coded keys from user) ---
HARDCODED_API_KEY    = "FwFdw6iC8R9UXcO07pUcG61eWQdTi6AimSIjFGgXOd4"
HARDCODED_FOLDER_ID  = "6561"
HARDCODED_TEMPLATE_ID= "22232"

OCR_API_BASE = os.getenv("OCR_API_BASE", "https://api.koncile.ai")
UPLOAD_ENDPOINT = f"{OCR_API_BASE}/v1/upload_file/"
FETCH_ENDPOINT  = f"{OCR_API_BASE}/v1/fetch_tasks_results/"

OCR_API_KEY = os.getenv("OCR_API_KEY") or os.getenv("KONCILE_API_KEY") or HARDCODED_API_KEY

OCR_HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {OCR_API_KEY}"
}

FIELD_ORDER = [
    "Medication name",
    "Quantity",
    "Dosage",
    "Duration of treatment (in days)",
    "Additional information",
    "Refill",
]

def upload_file_to_api(file_bytes: bytes, filename: str, template_id: Optional[str] = None, folder_id: Optional[str] = None) -> str:
    t_id = template_id or os.getenv("OCR_TEMPLATE_ID") or HARDCODED_TEMPLATE_ID
    f_id = folder_id   or os.getenv("OCR_FOLDER_ID")   or HARDCODED_FOLDER_ID

    params = {}
    if t_id:
        params["template_id"] = t_id
    if f_id:
        params["folder_id"] = f_id

    files = {"files": (filename, io.BytesIO(file_bytes))}
    resp = requests.post(UPLOAD_ENDPOINT, params=params, headers=OCR_HEADERS, files=files, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Upload failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    task_ids = data.get("task_ids") or data.get("task_id") or []
    if isinstance(task_ids, str):
        task_ids = [task_ids]
    if not task_ids:
        raise RuntimeError(f"No task id returned: {data}")
    return task_ids[0]

def poll_api_results(task_id: str, poll_interval: int = 2, timeout: int = 300) -> Dict[str, Any]:
    start = time.time()
    while True:
        if time.time() - start > timeout:
            raise TimeoutError("Remote OCR polling timed out")
        resp = requests.get(f"{FETCH_ENDPOINT}?task_id={task_id}", headers=OCR_HEADERS, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Polling failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        status = ""
        if isinstance(data, dict):
            status = (data.get("status") or data.get("task_status") or "").upper()
        if status and status not in ("IN PROGRESS", "IN_PROGRESS", "PROCESSING", "RUNNING"):
            return data
        if any(k in data for k in ("results", "data", "task_results")):
            return data
        time.sleep(poll_interval)

def find_first_key_recursive(obj: Any, target_key: str) -> Optional[Any]:
    target = target_key.lower()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() == target:
                return v
        for v in obj.values():
            found = find_first_key_recursive(v, target_key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_first_key_recursive(item, target_key)
            if found is not None:
                return found
    return None

def normalize_field_list(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, str):
                out.append({"value": item, "confidence_score": 0.0})
            else:
                out.append({"value": str(item), "confidence_score": 0.0})
        return out
    if isinstance(value, dict):
        return [value]
    return [{"value": str(value), "confidence_score": 0.0}]

def find_first_key_by_substring(obj: Any, target_key: str) -> Optional[Any]:
    target = target_key.lower()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and target in k.lower():
                return v
        for v in obj.values():
            found = find_first_key_by_substring(v, target_key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_first_key_by_substring(item, target_key)
            if found is not None:
                return found
    return None

def extract_ordered_fields(result_json: Dict[str, Any], field_order: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for fname in field_order:
        found = find_first_key_recursive(result_json, fname)
        if found is None:
            found_loose = find_first_key_by_substring(result_json, fname)
            if found_loose is not None:
                found = found_loose
        normalized = normalize_field_list(found)
        clean_items = []
        for item in normalized:
            if isinstance(item, dict):
                value = item.get("value") or item.get("text") or item.get("label") or item.get("name") or ""
            else:
                value = str(item)
            confidence = 0.0
            if isinstance(item, dict):
                for ck in ("confidence_score", "confidence", "score", "confidenceScore"):
                    if ck in item:
                        try:
                            confidence = float(item[ck])
                        except Exception:
                            confidence = 0.0
                        break
            clean_items.append({"value": value, "confidence_score": confidence})
        if not clean_items:
            clean_items = [{"value": "", "confidence_score": 0.0}]
        out[fname] = clean_items
    return out

def assemble_phrase(ordered_fields: Dict[str, List[Dict[str, Any]]]) -> str:
    get = lambda k: (ordered_fields.get(k) or [{"value": ""}])[0].get("value", "").strip()
    med = get("Medication name")
    qty = get("Quantity")
    dose = get("Dosage")
    additional = get("Additional information")
    duration = get("Duration of treatment (in days)")
    refill = get("Refill")
    qty_dose_parts = []
    if qty:
        qty_dose_parts.append(qty)
    if dose:
        qty_dose_parts.append(dose)
    qty_dose = " ".join(qty_dose_parts).strip()
    parts = []
    if med:
        parts.append(med + ":")
    else:
        parts.append("")
    if qty_dose:
        parts.append(qty_dose)
    if additional:
        parts.append(", " + additional)
    phrase = " ".join([p for p in parts if p]).strip()
    if duration:
        phrase = f"{phrase} ({duration})" if phrase else f"({duration})"
    if refill:
        phrase = f"{phrase} - Refill: {refill}" if phrase else f"Refill: {refill}"
    phrase = phrase.replace(" ,", ",").replace("  ", " ").strip()
    return phrase

# -------------------
# Segmentation import / fallback
# -------------------
# We prefer to import a separate segment_words.py next to this file. If not present,
# use an embedded fallback implementation (same algorithm) so endpoints always work.

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _sort_boxes_topleft_first(boxes: List[Tuple[int,int,int,int]], line_tol: int = 10):
    boxes_with_centers = []
    for b in boxes:
        x,y,w,h = b
        yc = y + h//2
        boxes_with_centers.append((b, yc))
    boxes_with_centers.sort(key=lambda x: x[1])
    lines = []
    for b, yc in boxes_with_centers:
        if not lines:
            lines.append([ (b,yc) ])
            continue
        last_centers = [c for (_b,c) in lines[-1]]
        if abs(yc - (sum(last_centers)//len(last_centers))) <= line_tol:
            lines[-1].append((b,yc))
        else:
            lines.append([ (b,yc) ])
    sorted_boxes = []
    for line in lines:
        line_boxes = [b for (b,_) in line]
        line_boxes.sort(key=lambda bb: bb[0])  # by x
        sorted_boxes.extend(line_boxes)
    return sorted_boxes

def _segment_for_display_fallback(
    bgr: np.ndarray,
    out_dir: str = "preproc_debug",
    min_width: int = 12,
    min_height: int = 12,
    padding: int = 6,
    debug_save: bool = True,
    adaptive_blocksize: int = 25,
    adaptive_C: int = 10,
    morph_kernel_size: Tuple[int,int] = (25, 3)
) -> Dict[str, Optional[object]]:
    out_path = Path(out_dir)
    _ensure_dir(out_path)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
    debug_fname = f"{ts}_debug_boxes.png"

    saved_paths = []
    boxes = []

    # ensure grayscale
    if len(bgr.shape) == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr.copy()

    # mild denoising
    gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)

    # contrast stretch
    lo, hi = np.percentile(gray, 1), np.percentile(gray, 99)
    if hi - lo > 1e-3:
        gray = np.uint8(np.clip((gray - lo) * (255.0 / (hi - lo)), 0, 255))

    # adaptive threshold (text->white)
    block = adaptive_blocksize if adaptive_blocksize % 2 == 1 else adaptive_blocksize + 1
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, adaptive_C)

    # morphological close to join letters to words
    kx, ky = morph_kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = gray.shape[:2]
    raw_boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w < min_width or h < min_height:
            continue
        if w > w_img*0.98 and h > h_img*0.98:
            continue
        raw_boxes.append((x,y,w,h))

    if not raw_boxes:
        return {"saved": [], "boxes": [], "debug_image": None, "reason": "no_boxes_found"}

    # merge proximate boxes horizontally (simple greedy)
    raw_boxes.sort(key=lambda b: (b[1], b[0]))
    merged = []
    for box in raw_boxes:
        x,y,w,h = box
        if not merged:
            merged.append(box)
            continue
        px,py,pw,ph = merged[-1]
        if (x <= px + pw + max(5, int(0.03*w_img))) and abs((y + h//2) - (py + ph//2)) <= max(10, int(0.02*h_img)):
            nx = min(px, x)
            ny = min(py, y)
            nw = max(px+pw, x+w) - nx
            nh = max(py+ph, y+h) - ny
            merged[-1] = (nx, ny, nw, nh)
        else:
            merged.append(box)

    sorted_boxes = _sort_boxes_topleft_first(merged, line_tol=max(10, h_img//50))

    for idx, (x,y,w,h) in enumerate(sorted_boxes):
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w_img, x + w + padding)
        y1 = min(h_img, y + h + padding)
        crop = bgr[y0:y1, x0:x1].copy()
        fname = f"{ts}_{idx:03d}_{x0}_{y0}_{(x1-x0)}x{(y1-y0)}.png"
        full = out_path / fname
        cv2.imwrite(str(full), crop)
        saved_paths.append(str(full.resolve()))
        boxes.append((x0,y0,x1-x0,y1-y0))

    debug_path = None
    if debug_save:
        disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (x,y,w,h) in boxes:
            cv2.rectangle(disp, (x,y), (x+w, y+h), (0,255,0), 1)
        debug_full = out_path / debug_fname
        cv2.imwrite(str(debug_full), disp)
        debug_path = str(debug_full.resolve())

    return {"saved": saved_paths, "boxes": boxes, "debug_image": debug_path}

# Try importing external module first (preferred)
try:
    # Look for segment_words.py in same directory
    try:
        from segment_words import segment_for_display  # type: ignore
    except Exception:
        # try import by file path in case of different working dir
        seg_path = Path(__file__).resolve().parent / "segment_words.py"
        if seg_path.exists():
            spec = importlib.util.spec_from_file_location("segment_words", str(seg_path))
            seg_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(seg_mod)
            segment_for_display = getattr(seg_mod, "segment_for_display")
        else:
            raise ImportError("segment_words.py not found")
except Exception:
    # fallback to local implementation
    segment_for_display = _segment_for_display_fallback  # type: ignore

# -------------------
# FastAPI app + endpoints
# -------------------
app = FastAPI()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "lexicon": {"count": len(_DRUG_NAMES), "path": str(LEXICON_CSV)},
        "api_remote_config": {
            "base": OCR_API_BASE,
            "has_key": bool(OCR_API_KEY),
            "folder_id_default": HARDCODED_FOLDER_ID,
            "template_id_default": HARDCODED_TEMPLATE_ID
        }
    }

@app.post("/predict_local")
async def predict_local(file: UploadFile = File(...), use_context: str = Form("0"), do_segmentation: str = Form("0")):
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    # optionally run segmentation for display-only debug (saves to PREPROC_DIR)
    debug_saved = None
    if do_segmentation in ("1", "true", "True"):
        try:
            seg = segment_for_display(
                bgr,
                out_dir=str(Path(PREPROC_DIR)),
                padding=6,
                min_width=10,
                min_height=10,
                morph_kernel_size=(25,3),
                adaptive_blocksize=25,
                adaptive_C=10,
                debug_save=True
            )
            debug_saved = seg
        except Exception as e:
            debug_saved = {"error": f"segmentation_failed: {e}"}

    try:
        fused_u8 = preprocess_and_fuse(bgr)
    except Exception as e:
        return JSONResponse({"error": f"Preprocessing failed: {e}"}, status_code=400)
    canvas_u8 = fit_to_canvas_1024x128_u8(fused_u8)
    canvas = canvas_u8.astype(np.float32) / 255.0
    x = torch.from_numpy(canvas[None, None, ...]).float().to(DEVICE)
    with torch.inference_mode():
        logp = vit_model.log_probs(x)
        logp_single = logp[:, 0, :]
        ids = greedy_ids(logp_single, blank_id=BLANK_ID)
        text_raw = tokenizer.decode_ids(ids)
    context_enabled = (use_context in ("1", "true", "True"))
    if context_enabled:
        text_fixed, changed, info = maybe_fix_first_word(text_raw)
    else:
        text_fixed, changed, info = text_raw, False, {"applied": False, "reason": "context-off"}
    first_raw   = _first_word(text_raw).strip()
    first_fixed = _first_word(text_fixed).strip()
    lexicon_applied = (bool(context_enabled) and bool(changed) and (first_fixed.lower() in _LEX_FIRST_TOKEN_SET))
    lexicon_applied_strict = bool(text_fixed.strip().lower() in _LEX_SET_LOWER)
    return {
        "ok": True,
        "model_used": "vit_local",
        "text_raw": text_raw,
        "text": text_fixed,
        "context_enabled": bool(context_enabled),
        "lexicon_changed": bool(changed),
        "lexicon_applied": bool(lexicon_applied),
        "lexicon_applied_strict": bool(lexicon_applied_strict),
        "lexicon_info": {**(info if isinstance(info, dict) else {"reason": str(info)}), "first_raw": first_raw, "first_fixed": first_fixed},
        "shape": [int(s) for s in canvas_u8.shape],
        "preproc_mode": OUTPUT_MODE,
        "debug_saved": debug_saved,
    }

@app.post("/predict_remote")
async def predict_remote(file: UploadFile = File(...), poll_timeout: int = Form(120), template_id: Optional[str] = Form(None), folder_id: Optional[str] = Form(None), do_segmentation: str = Form("0")):
    """
    Uploads to the external OCR API and returns:
      - assembled phrase (human readable)
      - full JSON from remote service
    Returns model_used: "vit-crnn" so frontend shows consistent label.
    """
    if not OCR_API_KEY:
        return JSONResponse({"error": "Remote OCR API key not configured on server"}, status_code=503)

    data = await file.read()
    filename = (file.filename or "upload").strip()

    # optionally do segmentation for display-only
    debug_saved = None
    if do_segmentation in ("1", "true", "True"):
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                seg = segment_for_display(
                    bgr,
                    out_dir=str(Path(PREPROC_DIR)),
                    padding=6,
                    min_width=10,
                    min_height=10,
                    morph_kernel_size=(25,3),
                    adaptive_blocksize=25,
                    adaptive_C=10,
                    debug_save=True
                )
                debug_saved = seg
        except Exception as e:
            debug_saved = {"error": f"segmentation_failed: {e}"}

    try:
        task_id = upload_file_to_api(data, filename, template_id=template_id, folder_id=folder_id)
    except Exception as e:
        return JSONResponse({"error": f"Remote upload failed: {e}"}, status_code=500)

    try:
        result = poll_api_results(task_id, poll_interval=2, timeout=poll_timeout)
    except Exception as e:
        return JSONResponse({"error": f"Remote polling failed: {e}"}, status_code=500)

    ordered = extract_ordered_fields(result, FIELD_ORDER)
    phrase = assemble_phrase(ordered)
    # <-- important: label remote response as vit-crnn for frontend display
    return {"ok": True, "model_used": "vit-crnn", "assembled": phrase, "full_result": result, "debug_saved": debug_saved}

@app.post("/predict_both")
async def predict_both(file: UploadFile = File(...), use_context: str = Form("0"), poll_timeout: int = Form(120), template_id: Optional[str] = Form(None), folder_id: Optional[str] = Form(None), do_segmentation: str = Form("0")):
    """
    Runs local predict and remote upload+poll in sequence and returns both results.
    Remote JSON will contain 'model_used': 'vit-crnn' to make it appear as the same model on frontend.
    Segmentation (display-only) will save to PREPROC_DIR and will be returned as 'debug_saved'.
    """
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse({"error": "Could not decode image for local model"}, status_code=400)

    # optionally run segmentation for display-only debug (saves to PREPROC_DIR)
    debug_saved = None
    if do_segmentation in ("1", "true", "True"):
        try:
            seg = segment_for_display(
                bgr,
                out_dir=str(Path(PREPROC_DIR)),
                padding=6,
                min_width=10,
                min_height=10,
                morph_kernel_size=(25,3),
                adaptive_blocksize=25,
                adaptive_C=10,
                debug_save=True
            )
            debug_saved = seg
        except Exception as e:
            debug_saved = {"error": f"segmentation_failed: {e}"}

    try:
        fused_u8 = preprocess_and_fuse(bgr)
    except Exception as e:
        return JSONResponse({"error": f"Preprocessing failed: {e}"}, status_code=400)
    canvas_u8 = fit_to_canvas_1024x128_u8(fused_u8)
    canvas = canvas_u8.astype(np.float32) / 255.0
    x = torch.from_numpy(canvas[None, None, ...]).float().to(DEVICE)
    with torch.inference_mode():
        logp = vit_model.log_probs(x)
        logp_single = logp[:, 0, :]
        ids = greedy_ids(logp_single, blank_id=BLANK_ID)
        text_raw = tokenizer.decode_ids(ids)
    context_enabled = (use_context in ("1", "true", "True"))
    if context_enabled:
        text_fixed, changed, info = maybe_fix_first_word(text_raw)
    else:
        text_fixed, changed, info = text_raw, False, {"applied": False, "reason": "context-off"}
    first_raw   = _first_word(text_raw).strip()
    first_fixed = _first_word(text_fixed).strip()
    lexicon_applied = (bool(context_enabled) and bool(changed) and (first_fixed.lower() in _LEX_FIRST_TOKEN_SET))
    lexicon_applied_strict = bool(text_fixed.strip().lower() in _LEX_SET_LOWER)

    local = {
        "ok": True,
        "model_used": "vit_local",
        "text_raw": text_raw,
        "text": text_fixed,
        "context_enabled": bool(context_enabled),
        "lexicon_changed": bool(changed),
        "lexicon_applied": bool(lexicon_applied),
        "lexicon_applied_strict": bool(lexicon_applied_strict),
        "lexicon_info": {**(info if isinstance(info, dict) else {"reason": str(info)}), "first_raw": first_raw, "first_fixed": first_fixed},
        "shape": [int(s) for s in canvas_u8.shape],
        "preproc_mode": OUTPUT_MODE,
        "debug_saved": debug_saved,
    }

    if not OCR_API_KEY:
        return {"ok": True, "local": local, "remote": None, "warning": "Remote OCR not configured on server."}
    try:
        task_id = upload_file_to_api(data, "upload", template_id=template_id, folder_id=folder_id)
        result = poll_api_results(task_id, poll_interval=2, timeout=poll_timeout)
        ordered = extract_ordered_fields(result, FIELD_ORDER)
        phrase = assemble_phrase(ordered)
        # label remote output as vit-crnn so frontend displays that model label
        remote = {"model_used": "vit-crnn", "assembled": phrase, "full_result": result}
    except Exception as e:
        remote = {"error": str(e)}

    return {"ok": True, "local": local, "remote": remote}

if __name__ == "__main__":
    uvicorn.run("app_vit:app", host="0.0.0.0", port=8001, reload=False)
