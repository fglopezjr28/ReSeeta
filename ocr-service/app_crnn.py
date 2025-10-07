# app_crnn.py
import os, numpy as np, cv2, torch, importlib.util
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

# -----------------------------
# Model / tokenization constants (same as app_vit.py)
# -----------------------------
IMG_H, IMG_W = 128, 1024
BLANK_ID = 0

import string
charset_base = string.printable[:95]  # 95 printable ASCII
VOCAB_SIZE = len(charset_base) + 1    # +1 for blank

# -----------------------------
# Tokenizer (CTC)  (same as app_vit.py)
# -----------------------------
class SimpleTokenizer:
    def __init__(self, charset: str, blank_id: int = 0):
        self.i2c = {i+1: c for i, c in enumerate(list(charset))}
        self.blank = blank_id
    def decode_ids(self, ids: List[int]) -> str:
        return "".join(self.i2c.get(int(i), "") for i in ids if int(i) != self.blank)

tokenizer = SimpleTokenizer(charset_base, blank_id=BLANK_ID)

# -----------------------------
# Dynamic import helper
# -----------------------------
def import_from_path(py_path: str, module_name: str = "baseline_model"):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# -----------------------------
# Greedy CTC collapse (T,B,C) -> ids  (mirrors app_vit.py logic)
# -----------------------------
def greedy_ids_T_B_C(logp_T_B_C: torch.Tensor, blank_id: int = 0) -> List[int]:
    logp_T_C = logp_T_B_C[:, 0, :]               # (T,C)
    ids = logp_T_C.argmax(dim=-1).cpu().numpy()  # (T,)
    out, prev = [], None
    for k in ids:
        k = int(k)
        if k != blank_id and k != prev:
            out.append(k)
        prev = k
    return out

# ==========================================================
#     PREPROCESSING — EXACT COPY OF app_vit.py BEHAVIOR
#  Noise Reduction -> Normalization -> (optional) Gaussian
#  -> Canny -> fuse (edges_only/overlay/norm_only) -> invert
#  -> letterbox (NEAREST) to 1024x128
# ==========================================================
# env-controlled knobs (same names as app_vit.py)
DENOISE_STRENGTH = int(os.getenv("PP_DENOISE_STRENGTH", 7))
DENOISE_TEMPLATE = int(os.getenv("PP_DENOISE_TEMPLATE", 7))
DENOISE_SEARCH   = int(os.getenv("PP_DENOISE_SEARCH",   21))

GAUSS_BLUR_KSIZE = int(os.getenv("PP_GAUSS_KSIZE", 3))  # odd >=3 to enable
GAUSS_BLUR_SIGMA = int(os.getenv("PP_GAUSS_SIGMA", 0))

CANNY_T1       = int(os.getenv("PP_CANNY_T1", 50))
CANNY_T2       = int(os.getenv("PP_CANNY_T2", 150))
CANNY_APERTURE = int(os.getenv("PP_CANNY_APERTURE", 3))
USE_L2GRAD     = os.getenv("PP_CANNY_L2", "1") not in ("0","false","False")

OUTPUT_MODE    = os.getenv("PP_OUTPUT_MODE", "edges_only").strip().lower()
INVERT_OUTPUT  = os.getenv("PP_INVERT", "1") not in ("0","false","False")

def _normalize_0_255(img_gray_u8: np.ndarray) -> np.ndarray:
    img = img_gray_u8.astype(np.float32)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi - lo < 1e-3:
        return np.uint8(np.clip(img, 0, 255))
    img = (img - lo) * (255.0 / (hi - lo))
    return np.uint8(np.clip(img, 0, 255))

def preprocess_and_fuse(img_bgr: np.ndarray) -> np.ndarray:
    # 1) Gray
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 2) Denoise
    den = cv2.fastNlMeansDenoising(
        gray, None,
        h=DENOISE_STRENGTH,
        templateWindowSize=DENOISE_TEMPLATE,
        searchWindowSize=DENOISE_SEARCH
    )
    # 3) Normalize
    norm = _normalize_0_255(den)
    # 4) Optional blur
    blur = norm
    if GAUSS_BLUR_KSIZE and GAUSS_BLUR_KSIZE >= 3 and GAUSS_BLUR_KSIZE % 2 == 1:
        blur = cv2.GaussianBlur(norm, (GAUSS_BLUR_KSIZE, GAUSS_BLUR_KSIZE), GAUSS_BLUR_SIGMA)
    # 5) Canny
    edges = cv2.Canny(blur, CANNY_T1, CANNY_T2, apertureSize=CANNY_APERTURE, L2gradient=USE_L2GRAD)
    # 6) Fuse
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
    return final  # uint8, bg white, ink black when INVERT_OUTPUT=True

def fit_to_canvas_1024x128_u8(img_u8: np.ndarray) -> np.ndarray:
    """Aspect-fit onto white 1024x128 using NEAREST (keeps Canny edges crisp)."""
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
# Load CRNN baseline model
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CRNN_MODEL_PY = os.environ.get("CRNN_MODEL_PY", "baseline_model.py")
CRNN_WEIGHTS  = os.environ.get("CRNN_WEIGHTS",  "CRNN_weights.pth")

m = import_from_path(CRNN_MODEL_PY, module_name="baseline_model")

# Use your baseline config (align with infer_baseline.py)
try:
    CRNNConfig = m.CRNNConfig
except AttributeError:
    raise RuntimeError("baseline_model.py must define CRNNConfig")

cfg = CRNNConfig(
    in_ch=1,
    num_classes=VOCAB_SIZE,   # 96 (blank=0 + 95 printable)
    feat_dim=320,
    use_proj_1x1=True,
    pool_type="avg",
    lstm_hidden=128,
    lstm_layers=2,
    lstm_dropout=0.2,
)

try:
    CRNNBaseline = m.CRNNBaseline
except AttributeError:
    raise RuntimeError("baseline_model.py must define CRNNBaseline")

model = CRNNBaseline(cfg)
with torch.no_grad():
    _ = model.forward(torch.zeros(1, 1, IMG_H, IMG_W))
if os.path.isfile(CRNN_WEIGHTS):
    sd = torch.load(CRNN_WEIGHTS, map_location="cpu")
    state = sd.get("model", sd) if isinstance(sd, dict) else sd
    missing, unexpected = model.load_state_dict(state, strict=False)
else:
    missing, unexpected = [], []
model.to(DEVICE).eval()

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI()

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": str(DEVICE),
        "weights_found": os.path.isfile(CRNN_WEIGHTS),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
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
    # 1) Decode image as BGR
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    # 2) Preprocess — EXACTLY like app_vit.py
    try:
        fused_u8  = preprocess_and_fuse(bgr)          # uint8
        canvas_u8 = fit_to_canvas_1024x128_u8(fused_u8)
    except Exception as e:
        return JSONResponse({"error": f"Preprocessing failed: {e}"}, status_code=400)

    # 3) To float tensor (0..1), shape (1,1,H,W)
    x = (canvas_u8.astype(np.float32) / 255.0)[None, None, ...]
    X = torch.from_numpy(x).to(DEVICE)

    # 4) Predict
    with torch.inference_mode():
        logp = model.log_probs(X)  # (T,B,C)
        ids  = greedy_ids_T_B_C(logp, blank_id=BLANK_ID)
        text = tokenizer.decode_ids(ids)

    return {
        "text": text,
        "shape": [int(s) for s in canvas_u8.shape],
        "preproc_mode": OUTPUT_MODE,
        "inverted": bool(INVERT_OUTPUT),
    }

if __name__ == "__main__":
    uvicorn.run("app_crnn:app", host="0.0.0.0", port=8002, reload=False)
