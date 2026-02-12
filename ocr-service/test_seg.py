# run_one_seg.py
from pathlib import Path
import cv2, sys
from segment_words import segment_for_display

# change this to your path (already provided)
img_path = Path(r"C:\Users\ferna\Desktop\4th Year\1st Sem\CS TW2\another approach\two-lines.jpg")

if not img_path.exists():
    print("ERROR: image not found:", img_path)
    sys.exit(1)

print("Loading image:", img_path)
bgr = cv2.imread(str(img_path))
if bgr is None:
    print("ERROR: cv2 failed to read the image.")
    sys.exit(1)

out_dir = Path("preproc_debug")
print("Output folder (will be created if needed):", out_dir.resolve())

res = segment_for_display(
    bgr,
    out_dir=str(out_dir),
    padding=6,
    min_width=10,
    min_height=10,
    morph_kernel_size=(25,3),
    adaptive_blocksize=25,
    adaptive_C=10,
    debug_save=True
)

print("SEGMENTATION RESULT:")
for k,v in res.items():
    if isinstance(v, list):
        print(f"  {k}: {len(v)} items")
    else:
        print(f"  {k}: {v}")

if res.get("saved"):
    print("\nSaved crop files:")
    for p in res["saved"]:
        print("  ", p)
if res.get("debug_image"):
    print("\nDebug image:", res["debug_image"])
    print("Open it to inspect the green boxes (or copy path into explorer).")
