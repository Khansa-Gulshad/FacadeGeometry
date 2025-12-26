# process_data.py
import os
os.environ['USE_PYGEOS'] = '0'

import io
import csv
import time
import math
import argparse
import requests
import numpy as np

import geopandas as gpd
import torch
from io import BytesIO
from PIL import Image, ImageFile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# use your 3-class helpers from segmentation.py
from modules.segmentation import (
    save_full_color, save_three_color, remap_to_three, save_full_overlay, save_rgb,
    save_three_class_mask, save_three_class_npz   # <-- add these two
)
import modules.config as cfg

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# FOLDERS
# =========================
def prepare_folders(city: str):
    base = os.path.join(cfg.PROJECT_DIR, cfg.city_to_dir(city))
    for sub in ["seg", "seg_3class", "seg_3class_vis", "seg_full_vis", "seg_full_overlay", "seg_qa", "sample_images", "save_rgb"]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)

# =========================
# MODEL LOADING
# =========================
def get_models():
    """
    Load Mask2Former (Cityscapes) on GPU if available.
    """
    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-large-cityscapes-semantic"
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-cityscapes-semantic"
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return processor, model


def segment_image(image_pil, processor, model):
    """
    Run semantic segmentation and return torch tensor (H,W) int labels.
    """
    inputs = processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            outputs = model(**inputs)
            seg = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image_pil.size[::-1]]
            )[0].to('cpu')
        else:
            outputs = model(**inputs)
            seg = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image_pil.size[::-1]]
            )[0]
    return seg  # torch (H,W)


# =========================
# STREET VIEW FETCH (DIRECT)
# =========================
def fetch_gsv_image_by_location(
    lat, lon, heading, pitch=6, fov=70, size="640x640",
    api_key=None, retries=2, backoff=1.6, timeout=20, radius=300, outdoor_only=True
):
    """
    Robust Street View fetch:
      1) call Metadata API to get nearest pano_id (within `radius` meters)
      2) fetch the image with pano_id + heading/pitch/fov
    Raises RuntimeError with a clear message on failure.
    """
    assert api_key, "GSV API key required"
    src = "&source=outdoor" if outdoor_only else ""

    # 1) Metadata: nearest pano_id
    meta_url = (
        "https://maps.googleapis.com/maps/api/streetview/metadata"
        f"?location={lat},{lon}&radius={radius}{src}&key={api_key}"
    )
    mr = requests.get(meta_url, timeout=timeout)
    try:
        meta = mr.json()
    except Exception:
        raise RuntimeError(f"Metadata HTTP {mr.status_code}: {mr.text[:160]}")
    status = meta.get("status", "UNKNOWN")
    if status != "OK":
        raise RuntimeError(f"Metadata status {status}: {meta.get('error_message') or ''}".strip())
    pano_id = meta.get("pano_id")
    if not pano_id:
        raise RuntimeError("Metadata OK but missing pano_id")

    # 2) Image by pano
    url = (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size={size}&pano={pano_id}&heading={heading}&pitch={pitch}&fov={fov}{src}&key={api_key}"
    )

    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            ct = resp.headers.get("content-type", "")
            if resp.status_code != 200 or ("image" not in ct):
                # Google sometimes returns HTML/JSON bodies with 200; surface details:
                raise RuntimeError(f"Image fetch HTTP {resp.status_code}, CT={ct}, body={resp.text[:160]}")
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            last_err = e
            time.sleep(backoff ** attempt)

    raise RuntimeError(f"Image fetch failed after retries: {last_err}")


# =========================
# SINGLE VIEW →CLASS MASK
# =========================
def process_facade_view(img_pil, processor, model):
    """
    Run segmentation on one frame.
    Returns:
      mask_full: uint8 (H,W) values 0..18
      mask3    : uint8 (H,W) values {0,1,2,3} with 1=building,2=sky,3=ground
    """
    seg_full = segment_image(img_pil, processor, model)  # torch (H,W)
    mask_full = seg_full.cpu().numpy().astype(np.uint8)
    mask3 = remap_to_three(mask_full).astype(np.uint8)
    return mask_full, mask3

# =========================
# PER-POINT RUNNER (two headings: road_angle ± 90°)
# =========================
def _round_heading(h):  # nicer filenames
    return int(round(h)) % 360

def download_facade_masks_for_point(
    row, city, access_token, processor, model,
    pitch_deg=6, fov_deg=70, size="640x640",
    save_sample=False
):
    """
    row: GeoDataFrame row with:
      - id
      - geometry (Point; x=lon, y=lat) in WGS84
      - road_angle (deg from North)
    Fetch two images: headings = road_angle ± 90°, segment, save masks (+ overlay).
    """
    lat, lon = row.geometry.y, row.geometry.x
    try:
        ra = float(row.road_angle)
        if math.isnan(ra):
            ra = 0.0
    except Exception:
        ra = 0.0
    views = [
        ("left",  (ra - 90) % 360),
        ("right", (ra + 90) % 360),
            ]

    records = []

    for side, h in views:
        image_id = f"{row.id}_{side}"

        try:
            img = fetch_gsv_image_by_location(
                lat, lon,
                heading=h,
                pitch=pitch_deg,
                fov=fov_deg,
                size=size,
                api_key=access_token
            )

            mask_full, mask3 = process_facade_view(img, processor, model)

            save_rgb(city, image_id, img)
            save_full_color(city, image_id, mask_full)
            save_three_color(city, image_id, mask3)
            save_three_class_mask(city, image_id, mask3)
            label_npz = save_three_class_npz(city, image_id, mask3)

            if save_sample:
                save_full_overlay(
                    city, image_id,
                    np.array(img), mask_full,
                    alpha=0.65, soften_sigma=0.8
                )

            records.append([image_id, label_npz, side, pitch_deg, fov_deg])

        except Exception as e:
            records.append([image_id, f"ERROR: {e}", side, pitch_deg, fov_deg])


# =========================
# BATCH DRIVER (uses *your* points with road_angle)
# =========================
def download_images_for_points(
    gdf, access_token, city,
    pitch_deg=6, fov_deg=70, size="640x640",
    save_sample=False, max_workers=1
):
    """
    Runs façade segmentation on a points GDF that ALREADY has 'road_angle'.
    gdf must be in WGS84 (EPSG:4326).
    """
    prepare_folders(city)
    processor, model = get_models()

    manifest = []
    max_workers = max(1, int(max_workers))  # single-GPU → usually keep 1

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for _, row in gdf.iterrows():
            futures.append(
                ex.submit(
                    download_facade_masks_for_point,
                    row, city, access_token, processor, model,
                    pitch_deg, fov_deg, size, save_sample
                )
            )

        for f in tqdm(as_completed(futures), total=len(futures), desc="Façade masks (±90°)"):
            try:
                recs = f.result()
                manifest.extend(recs)
            except Exception:
                manifest.append(["POINT_ERROR", "ERROR", None, pitch_deg, fov_deg])

    # write manifest
    out_dir = os.path.join(cfg.PROJECT_DIR, cfg.city_to_dir(city), "seg_3class_vis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "mask_path", "side", "pitch_deg", "fov_deg"])
        for row in manifest:
            w.writerow(row)

    return manifest


# =========================
# CLI (reads your existing points file; no building/sweeps)
# =========================
def main():
    ap = argparse.ArgumentParser(description="Façade 3-class segmentation from points with road_angle (±90° only)")
    ap.add_argument("--city", type=str, required=True, help="City name (for results path)")
    ap.add_argument("--api_key", type=str, default=os.getenv("GSV_API_KEY"), help="Google Street View API key")
    ap.add_argument("--points", type=str, required=True, help="Path to GeoPackage/GeoJSON with points (must have 'road_angle')")
    ap.add_argument("--layer", type=str, default=None, help="Layer name if reading from GeoPackage")
    ap.add_argument("--pitch", type=float, default=6.0, help="Camera pitch (deg)")
    ap.add_argument("--fov", type=float, default=70.0, help="Field of view (deg)")
    ap.add_argument("--size", type=str, default="640x640", help="Image size for Static API")
    ap.add_argument("--save_qa", action="store_true", help="Save overlay QA images")
    ap.add_argument("--workers", type=int, default=1, help="Thread workers (keep 1 for single GPU)")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Provide --api_key or set env GSV_API_KEY")

    # read your existing points (already created by road_network.py pipeline)
    gdf = gpd.read_file(args.points, layer=args.layer) if args.layer else gpd.read_file(args.points)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    if "road_angle" not in gdf.columns:
        raise SystemExit("The provided points file must contain 'road_angle'.")

    if "id" not in gdf.columns:
        gdf = gdf.reset_index(drop=True)
        gdf["id"] = np.arange(len(gdf), dtype=int)

    # run
    download_images_for_points(
        gdf=gdf,
        access_token=args.api_key,
        city=args.city,
        pitch_deg=args.pitch,
        fov_deg=args.fov,
        size=args.size,
        save_sample=args.save_qa,
        max_workers=args.workers
    )

if __name__ == "__main__":
    main()
