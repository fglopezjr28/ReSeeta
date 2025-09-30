# app.py
import io, os, math, numpy as np
from typing import List, Tuple
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import torch
import importlib.util
import cv2

IMG_H, IMG_W = 128, 1024
BLANK_ID = 0

import string
charset_base = string.printable[:95]
VOCAB_SIZE = len(charset_base) + 1

# ---- Tokenizer ----
class SimpleTokenizer:
    def __init__(self, charset: str, blank_id: int = 0):
        self.i2c = {i+1: c for i, c in enumerate(list(charset))}
        self.blank = blank_id
    def decode_ids(self, ids: List[int]) -> str:
        return "".join(self.i2c.get(int(i), "") for i in ids if int(i) != self.blank)

tokenizer = SimpleTokenizer(charset_base, blank_id=BLANK_ID)

# ---- Import model module ----
def import_from_path(py_path: str, module_name: str = "reseeta_model"):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ---- CTC decode ----
def greedy_ids(logp_T_C: torch.Tensor, blank_id: int = 0) -> List[int]:
    ids = logp_T_C.argmax(dim=-1).cpu().numpy()
    out, prev = [], None
    for k in ids:
        k = int(k)
        if k != blank_id and k != prev:
            out.append(k)
        prev = k
    return out

# ---- Image to 128x1024 (no preprocessing) ----
def make_canvas_1024x128(gray01: np.ndarray) -> np.ndarray:
    h0, w0 = gray01.shape
    if (h0, w0) == (IMG_H, IMG_W):
        return gray01
    scale = min(IMG_W / w0, IMG_H / h0)
    nw = max(1, int(round(w0 * scale)))
    nh = max(1, int(round(h0 * scale)))
    resized = cv2.resize(gray01, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.ones((IMG_H, IMG_W), np.float32)
    y0 = (IMG_H - nh) // 2
    x0 = (IMG_W - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

# ---- Globals: load once ----
MODEL_PY = os.environ.get("MODEL_PY", "reseeta_model.py")
WEIGHTS = os.environ.get("WEIGHTS", "ViT_CRNN_weights.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m = import_from_path(MODEL_PY)
cfg = m.ViTCRNNConfig(in_ch=1, num_classes=VOCAB_SIZE, patch_w=1, norm_first=True)
model = m.ViTCRNN(cfg)
with torch.no_grad():
    _ = model.forward(torch.zeros(1, 1, IMG_H, IMG_W))
sd = torch.load(WEIGHTS, map_location="cpu")
state = sd.get("model", sd) if isinstance(sd, dict) else sd
model.load_state_dict(state, strict=False)
model.to(DEVICE).eval()

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # read bytes → grayscale float [0,1], NO preprocessing
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)
    img = (img.astype(np.float32) / 255.0)

    # if your uploads are already 128x1024, this is effectively a no-op
    canvas = make_canvas_1024x128(img)
    x = torch.from_numpy(canvas[None, None, ...]).float().to(DEVICE)

    with torch.inference_mode():
        logp = model.log_probs(x)      # (T,B,C)
        logp_single = logp[:, 0, :]
        ids_g = greedy_ids(logp_single, blank_id=BLANK_ID)
        text = tokenizer.decode_ids(ids_g)

    return {"text": text, "shape": [int(x) for x in canvas.shape]}

if __name__ == "__main__":
    # Run: python app.py  (or use the uvicorn command below)
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)
