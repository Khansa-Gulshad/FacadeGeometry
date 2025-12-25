import os
import requests
import shutil
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Creates required folder structure
def prepare_folders(city):
    folder_names = [
        "points",
        "roads",
        "rgb_images",
        "temp_sv_images",
        "sv_images",
        # SAM temp outputs for each prompt
        "temp_seg_sky",
        "temp_seg_facade",
        "temp_seg_windows",
        "temp_seg_ground",
        # final label / visualization
        "seg_npz",
        "seg_vis",
    ]

    for folder_name in folder_names:
        prepare_folder(city, folder_name)

# Creates a folder if it doesn't exist yet
def prepare_folder(city, folder_name):
  dir_path = os.path.join("/mnt/project/pt01183/facade_results", city, folder_name)
  if not os.path.exists(dir_path):
      os.makedirs(dir_path)


# Takes an input path and deletes all files inside of that path
def delete_files(path):
  files = os.listdir(path)

  for file_name in files:
    file_path = os.path.join(path, file_name)
    os.truncate(file_path, 0)
    os.remove(file_path)


# Takes a TIF PIL image and returns the number of white pixels in it
def count_white_pixels(image):
  # change 1: robust white detection across modes (1/L/RGB/P)
  im = image
  if im.mode == "P":
    im = im.convert("L")
  if im.mode == "1":
    return sum(1 for p in im.getdata() if p == 255)
  elif im.mode == "L":
    return sum(1 for p in im.getdata() if p >= 250)
  else:
    im = im.convert("RGB")
    return sum(1 for r, g, b in im.getdata() if r >= 250 and g >= 250 and b >= 250)


# Finds all images (files) inside two directories and returns them as lists
def load_images(dir1, dir2):
  images1 = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
  images2 = [f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))]

  return images1, images2


# Moves all files from one directory to another directory
def move_files(source_dir, destination_dir):
  files = os.listdir(source_dir)

  for f in files:
    source_path = os.path.join(source_dir, f)
    destination_path = os.path.join(destination_dir, f)
    shutil.move(source_path, destination_path)


# Takes a list of images and creates segmentation masks in the output folders
def segment_images(sam, images, city, index, save_streetview):
    """
    - Saves SVI to temp folder
    - Runs SAM for 4 prompts:
        sky, building facade, windows/doors, ground
    - Builds a single label map:
        0: background
        1: ground
        2: facade
        3: windows/doors
        4: sky
    - Saves:
        seg_npz/{index}.npz  with 'seg'
        seg_vis/{index}.png  with colors
    """
    BASE = f"/mnt/project/pt01183/facade_results/{city}"

    temp_path = os.path.join(BASE, "temp_sv_images")
    sv_path   = os.path.join(BASE, "sv_images")
    npz_dir   = os.path.join(BASE, "seg_npz")
    vis_dir   = os.path.join(BASE, "seg_vis")

    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(sv_path, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # ---------------------------------------------------
    # 1. Save SVI(s) temporarily
    # ---------------------------------------------------
    for i, image in enumerate(images):
        temp_output_path = os.path.join(temp_path, f"{index}_streetview_{i}.png")
        image.save(temp_output_path)

        if save_streetview:
            sv_output = os.path.join(sv_path, f"{index}_streetview_{i}.png")
            image.save(sv_output)

    # list images we just wrote
    files = sorted([f for f in os.listdir(temp_path) if f.endswith(".png")])
    if not files:
        print(f"[SAM] No temp images for index {index}")
        return

    # ---------------------------------------------------
    # 2. SAM segmentation for 4 text prompts
    # ---------------------------------------------------
    prompt_to_folder = {
        "sky":                     "temp_seg_sky",
        "building facade":         "temp_seg_facade",
        "windows doors openings":  "temp_seg_windows",
        "ground road pavement":    "temp_seg_ground",
    }

    delete_files(os.path.join(BASE, folder))

    for prompt, folder in prompt_to_folder.items():
        out_dir = os.path.join(BASE, folder)
        os.makedirs(out_dir, exist_ok=True)

        sam.predict_batch(
            images=temp_path,
            out_dir=out_dir,
            text_prompt=prompt,
            box_threshold=0.24,
            text_threshold=0.24,
            merge=False
        )

    # ---------------------------------------------------
    # 3. Build final label map from the FIRST view
    #    (index_streetview_0.tif)
    # ---------------------------------------------------
    sample_img_path = os.path.join(temp_path, files[0])
    sample_img = Image.open(sample_img_path)
    W, H = sample_img.size  # PIL: (width, height)
    label = np.zeros((H, W), dtype=np.uint8)

    # label code → folder mapping
    class_map = {
        4: "temp_seg_sky",      # sky
        2: "temp_seg_facade",   # facade
        3: "temp_seg_windows",  # windows/doors
        1: "temp_seg_ground"    # ground
    }

    for lbl, folder_name in class_map.items():
        folder_path = os.path.join(BASE, folder_name)
        mask_file = os.path.join(folder_path, f"{index}_streetview_0.png")
        if os.path.exists(mask_file):
            mask_img = Image.open(mask_file).convert("L")
            mask = np.array(mask_img) > 128
            label[mask] = lbl

    # ---------------------------------------------------
    # 4. Save NPZ
    # ---------------------------------------------------
    npz_path = os.path.join(npz_dir, f"{index}.npz")
    np.savez(npz_path, seg=label)
    print(f"[SAM] Saved label NPZ → {npz_path}")

    # ---------------------------------------------------
    # 5. Save visualization PNG
    # ---------------------------------------------------
    color_map = {
        0: (0,   0,   0  ),  # background
        1: (153, 102, 51 ),  # ground
        2: (255, 255, 0  ),  # facade
        3: (0,   0,   255),  # windows/doors
        4: (0,   255, 255),  # sky
    }

    vis = np.zeros((H, W, 3), dtype=np.uint8)
    for lbl, rgb in color_map.items():
        vis[label == lbl] = rgb

    vis_path = os.path.join(vis_dir, f"{index}.png")
    Image.fromarray(vis).save(vis_path)
    print(f"[SAM] Saved seg vis PNG → {vis_path}")

    # ---------------------------------------------------
    # 6. Clean temp SVI
    # ---------------------------------------------------
    delete_files(temp_path)

