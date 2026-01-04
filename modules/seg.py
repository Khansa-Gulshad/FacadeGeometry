# seg.py
import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from modules.segmentation import (
    remap_to_three,
    save_three_class_mask,
    save_three_class_npz,
    save_three_color,
    save_full_overlay,
)
import modules.config as cfg

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
img_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
print(f"[INFO] Found {len(images)} images to segment")

for fn in tqdm(img_files, desc="Segmenting images"):
    image_id = fn.replace(".jpg", "")

    # 🔴 SKIP if already processed
    seg_npz_path = os.path.join(
        cfg.PROJECT_DIR,
        cfg.city_to_dir(CITY),
        "seg",
        f"{image_id}_seg.npz"
    )
    if os.path.exists(seg_npz_path):
        continue

    # Load image
    img = Image.open(os.path.join(IMG_DIR, fn)).convert("RGB")

    # Move inputs to GPU
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    seg_full = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[img.size[::-1]]
    )[0].cpu().numpy().astype("uint8")

    # Remap to 3-class
    mask3 = remap_to_three(seg_full)

    # Save outputs (OpenFacade-compatible)
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
