# seg.py
import os
import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch


import modules.config as cfg
from modules.segmentation import (
    remap_to_three,
    save_three_class_mask,
    save_three_class_npz,
    save_three_color,
    save_full_overlay,
)

assert torch.cuda.is_available(), "❌ CUDA not available"
device = torch.device("cuda")
print("✅ Using GPU:", torch.cuda.get_device_name(0))

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
CITY = "Gdańsk, Poland"
IMG_DIR = os.path.join(
    cfg.PROJECT_DIR,
    cfg.city_to_dir(CITY),
    "save_rgb",
    "imgs"
)

USE_QA_OVERLAY = False   # turn on later if needed

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-large-cityscapes-semantic"
)

model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-cityscapes-semantic"
).to(device)

model.eval()

# -------------------------------------------------
# SEGMENTATION LOOP
# -------------------------------------------------
images = sorted(fn for fn in os.listdir(IMG_DIR) if fn.endswith(".jpg"))
print(f"[INFO] Found {len(images)} images to segment")

for fn in tqdm(images, desc="Mask2Former segmentation"):
    image_id = fn.replace(".jpg", "")
    img_path = os.path.join(IMG_DIR, fn)

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Could not open {fn}: {e}")
        continue

    try:
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        seg_full = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[img.size[::-1]]
        )[0].cpu().numpy().astype(np.uint8)

        mask3 = remap_to_three(seg_full)

        # -------------------------------------------------
        # SAVE OUTPUTS (same structure as before)
        # -------------------------------------------------
        save_three_class_mask(CITY, image_id, mask3)
        save_three_class_npz(CITY, image_id, mask3)
        save_three_color(CITY, image_id, mask3)

        if USE_QA_OVERLAY:
            save_full_overlay(
                CITY,
                image_id,
                np.array(img),
                seg_full,
                alpha=0.65,
                soften_sigma=0.8
            )

    except Exception as e:
        print(f"[ERROR] Failed on {fn}: {e}")

print("[DONE] Segmentation complete.")
