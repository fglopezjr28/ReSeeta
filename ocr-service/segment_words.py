# segment_words.py
"""
segment_words.py

Standalone segmentation module.
- Provides `segment_for_display()` which segments an input BGR image, saves crops + debug image
  into the provided out_dir (default: ./preproc_debug) and returns a summary dict.
- Purely for visualization/debug; does not alter prediction inputs.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
import os

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
            lines.append([(b, yc)])
            continue
        last_centers = [c for (_b, c) in lines[-1]]
        if abs(yc - (sum(last_centers)//len(last_centers))) <= line_tol:
            lines[-1].append((b, yc))
        else:
            lines.append([(b, yc)])
    sorted_boxes = []
    for line in lines:
        line_boxes = [b for (b, _) in line]
        line_boxes.sort(key=lambda bb: bb[0])  # sort by x
        sorted_boxes.extend(line_boxes)
    return sorted_boxes

def segment_for_display(
    bgr: np.ndarray,
    out_dir: str = "preproc_debug",
    min_width: int = 12,
    min_height: int = 12,
    padding: int = 6,
    debug_save: bool = True,
    adaptive_blocksize: int = 25,
    adaptive_C: int = 10,
    morph_kernel_size: Tuple[int, int] = (25, 3)
) -> Dict[str, Optional[object]]:
    """
    Segment words from a BGR numpy image and save crops + debug image to out_dir.
    Returns dict:
      {
        "saved": [full_paths...],
        "boxes": [(x,y,w,h), ...],
        "debug_image": full_path or None,
        "reason": optional explanation
      }
    """
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
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, block, adaptive_C)

    # morphological close to join letters to words
    kx, ky = morph_kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = gray.shape[:2]
    raw_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_width or h < min_height:
            continue
        if w > w_img * 0.98 and h > h_img * 0.98:
            continue
        raw_boxes.append((x, y, w, h))

    if not raw_boxes:
        return {"saved": [], "boxes": [], "debug_image": None, "reason": "no_boxes_found"}

    # merge proximate boxes horizontally (simple greedy) with fixed min_gap = 6
    min_gap = 6  # fixed horizontal merge gap in pixels
    raw_boxes.sort(key=lambda b: (b[1], b[0]))
    merged = []
    for box in raw_boxes:
        x, y, w, h = box
        if not merged:
            merged.append(box)
            continue
        px, py, pw, ph = merged[-1]
        if (x <= px + pw + min_gap) and abs((y + h//2) - (py + ph//2)) <= max(10, int(0.02 * h_img)):
            nx = min(px, x)
            ny = min(py, y)
            nw = max(px + pw, x + w) - nx
            nh = max(py + ph, y + h) - ny
            merged[-1] = (nx, ny, nw, nh)
        else:
            merged.append(box)

    sorted_boxes = _sort_boxes_topleft_first(merged, line_tol=max(10, h_img // 50))

    for idx, (x, y, w, h) in enumerate(sorted_boxes):
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w_img, x + w + padding)
        y1 = min(h_img, y + h + padding)
        crop = bgr[y0:y1, x0:x1].copy()
        fname = f"{ts}_{idx:03d}_{x0}_{y0}_{(x1 - x0)}x{(y1 - y0)}.png"
        full = out_path / fname
        cv2.imwrite(str(full), crop)
        saved_paths.append(str(full.resolve()))
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    debug_path = None
    if debug_save:
        disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in boxes:
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 1)
        debug_full = out_path / debug_fname
        cv2.imwrite(str(debug_full), disp)
        debug_path = str(debug_full.resolve())

    return {"saved": saved_paths, "boxes": boxes, "debug_image": debug_path}
