# app_vit.py
import io, os, math, numpy as np
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import torch
import importlib.util
import cv2

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
        self.i2c = {i+1: c for i, c in enumerate(list(charset))}
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

# ==========================================================
#                 PREPROCESSING (from your notebook)
#    Noise Reduction -> Normalization -> Canny -> Invert
# ==========================================================
# Defaults mirror your script; tweak here or via env if desired
DENOISE_STRENGTH = int(os.getenv("PP_DENOISE_STRENGTH", 7))
DENOISE_TEMPLATE = int(os.getenv("PP_DENOISE_TEMPLATE", 7))
DENOISE_SEARCH   = int(os.getenv("PP_DENOISE_SEARCH",   21))

GAUSS_BLUR_KSIZE = int(os.getenv("PP_GAUSS_KSIZE", 3))   # must be odd; set 0/1/2 to disable
GAUSS_BLUR_SIGMA = int(os.getenv("PP_GAUSS_SIGMA", 0))

CANNY_T1       = int(os.getenv("PP_CANNY_T1", 50))
CANNY_T2       = int(os.getenv("PP_CANNY_T2", 150))
CANNY_APERTURE = int(os.getenv("PP_CANNY_APERTURE", 3))
USE_L2GRAD     = os.getenv("PP_CANNY_L2", "1") not in ("0","false","False")

# One fused output: edges_only | overlay | norm_only
OUTPUT_MODE    = os.getenv("PP_OUTPUT_MODE", "edges_only").strip().lower()
# invert so background=white, strokes=black (recommended for your model)
INVERT_OUTPUT  = os.getenv("PP_INVERT", "1") not in ("0","false","False")

def _normalize_0_255(img_gray_u8: np.ndarray) -> np.ndarray:
    """Percentile stretch to [0,255] uint8 (1..99th)."""
    img = img_gray_u8.astype(np.float32)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi - lo < 1e-3:
        return np.uint8(np.clip(img, 0, 255))
    img = (img - lo) * (255.0 / (hi - lo))
    return np.uint8(np.clip(img, 0, 255))

def preprocess_and_fuse(img_bgr: np.ndarray) -> np.ndarray:
    """
    Returns a SINGLE-channel uint8 image following your pipeline:
      BGR->gray -> fastNlMeansDenoising -> percentile normalize -> (optional GaussianBlur)
      -> Canny -> fuse (edges_only / overlay / norm_only) -> optional invert
    Output: uint8, background white (255) and ink black (0) when INVERT_OUTPUT=True.
    """
    # 1) Gray
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2) Denoise
    den = cv2.fastNlMeansDenoising(
        gray, None,
        h=DENOISE_STRENGTH,
        templateWindowSize=DENOISE_TEMPLATE,
        searchWindowSize=DENOISE_SEARCH
    )

    # 3) Normalize (percentile stretch)
    norm = _normalize_0_255(den)

    # 4) Optional blur before Canny (if ksize is a valid odd >=3)
    blur = norm
    if GAUSS_BLUR_KSIZE and GAUSS_BLUR_KSIZE >= 3 and GAUSS_BLUR_KSIZE % 2 == 1:
        blur = cv2.GaussianBlur(norm, (GAUSS_BLUR_KSIZE, GAUSS_BLUR_KSIZE), GAUSS_BLUR_SIGMA)

    # 5) Canny (edges are white on black)
    edges = cv2.Canny(blur, CANNY_T1, CANNY_T2, apertureSize=CANNY_APERTURE, L2gradient=USE_L2GRAD)

    # 6) Fuse to ONE image
    if OUTPUT_MODE == "edges_only":
        final = edges  # white edges on black
        if INVERT_OUTPUT:
            # → bg white (255), ink black (0)
            final = cv2.bitwise_not(final)

    elif OUTPUT_MODE == "overlay":
        if INVERT_OUTPUT:
            # Force white bg + black edges only (no grayscale bg)
            final = np.full_like(norm, 255, dtype=np.uint8)
            final[edges > 0] = 0
        else:
            # White edges over normalized background
            final = norm.copy()
            final[edges > 0] = 255

    elif OUTPUT_MODE == "norm_only":
        final = norm
        if INVERT_OUTPUT:
            final = cv2.bitwise_not(final)

    else:
        raise ValueError(f"Unknown OUTPUT_MODE: {OUTPUT_MODE}")

    return final  # uint8
# (pipeline adapted from your preprocessing block)  # :contentReference[oaicite:1]{index=1}

# -----------------------------
# Fit to 1024x128 canvas (keep aspect)
# -----------------------------
def fit_to_canvas_1024x128_u8(img_u8: np.ndarray) -> np.ndarray:
    """Aspect-fit uint8 image onto a white 1024x128 canvas; keeps crisp edges."""
    h0, w0 = img_u8.shape
    if (h0, w0) == (IMG_H, IMG_W):
        return img_u8
    scale = min(IMG_W / w0, IMG_H / h0)
    nw = max(1, int(round(w0 * scale)))
    nh = max(1, int(round(h0 * scale)))
    # Use NEAREST to avoid softening edges from Canny
    resized = cv2.resize(img_u8, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((IMG_H, IMG_W), 255, np.uint8)  # white
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
model = m.ViTCRNN(cfg)
with torch.no_grad():
    _ = model.forward(torch.zeros(1, 1, IMG_H, IMG_W))
sd = torch.load(WEIGHTS, map_location="cpu")
state = sd.get("model", sd) if isinstance(sd, dict) else sd
model.load_state_dict(state, strict=False)
model.to(DEVICE).eval()

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
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
async def predict(file: UploadFile = File(...)):
    # 1) Read & decode as COLOR (BGR)
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    # 2) Preprocess (uint8 → single fused image)
    try:
        fused_u8 = preprocess_and_fuse(bgr)      # uint8, bg white, ink black
    except Exception as e:
        return JSONResponse({"error": f"Preprocessing failed: {e}"}, status_code=400)

    # 3) Fit to 1024×128 canvas (uint8)
    canvas_u8 = fit_to_canvas_1024x128_u8(fused_u8)  # uint8

    # 4) To float tensor (0..1), shape (1,1,H,W)
    canvas = canvas_u8.astype(np.float32) / 255.0
    x = torch.from_numpy(canvas[None, None, ...]).float().to(DEVICE)

    # 5) Predict
    with torch.inference_mode():
        logp = model.log_probs(x)      # (T,B,C) log-probs
        logp_single = logp[:, 0, :]
        ids = greedy_ids(logp_single, blank_id=BLANK_ID)
        text = tokenizer.decode_ids(ids)

    return {
        "text": text,
        "shape": [int(s) for s in canvas_u8.shape],   # [H,W]
        "preproc_mode": OUTPUT_MODE,
        "inverted": bool(INVERT_OUTPUT),
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)
