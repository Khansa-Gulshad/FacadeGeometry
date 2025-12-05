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


# Function adapted from iabrilvzqz
# GitHub: https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility-GSV
# Processes street view imagery (SVI) based on an image URL.
# Returns a list of processed images (1 for normal SVI, 2 for panoramic SVI)
def process_image(img_or_url, is_panoramic, road_angle):
    """
    Mapillary-style:
    - Accept a URL *or* a PIL.Image
    - If panoramic: crop bottom 20%, then extract the two perpendicular faces.
    - If not: just return the single image.
    """
    # NEW: accept PIL image as well as URL
    if isinstance(img_or_url, str):
        image = Image.open(requests.get(img_or_url, stream=True).raw)
    else:
        image = img_or_url  # PIL.Image.Image

    if is_panoramic:
        width, height = image.size
        # Mapillary logic: keep top 80% (crop out bottom 20%)
        cropped_height = int(height * 0.8)
        image = image.crop((0, 0, width, cropped_height))

        # Split pano into two perpendicular faces using road_angle
        left_face, right_face = get_perpendicular_images(image, road_angle)
        images = [left_face, right_face]
    else:
        images = [image]

    return images
# Takes a panoramic SVI and returns two images looking perpendicular from the road
def get_perpendicular_images(image, road_angle):
    width, height = image.size
    eighth_width = int(0.125 * width)

    # We want left and right facing images. Wrap around in case values are out of bounds
    wanted_angles = ((road_angle - 90) % 360, (road_angle + 90) % 360)

    faces = []
    original_image = image.copy()

    # We want 1/8th of the image before and after the wanted angle within the shot (1/4th total)
    for wanted_angle in wanted_angles:
        image = original_image.copy()

        # E.g. if wanted_angle is 10, the wanted shift is to fraction 0.0278  of the image on a 0-1 range
        wanted_fractional_axis = float(wanted_angle)/360.0
        wanted_axis= int(width * wanted_fractional_axis)

        left_max = max(wanted_axis - eighth_width, 0)
        right_max = min(wanted_axis + eighth_width, width)
        perpendicular_face = image.crop((left_max, 0, right_max, height))

        faces.append(perpendicular_face)

    # Return the left and right perpendicular face
    return faces


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

# Calculates weighted average ratio (WAR) of two perpendicular looking images
def calculate_WAR(width_A, height_A, ratio_A, width_B, height_B, ratio_B):
  size_A = width_A * height_A
  size_B = width_B * height_B

  # Calculate weighted ratio (WR) for each image
  WR_A = ratio_A * size_A
  WR_B = ratio_B * size_B

  # Calculate weighted average ratio (WAR)
  WR_total = WR_A + WR_B
  total_size = size_A + size_B

  WAR = WR_total / total_size

  return WAR


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
        temp_output_path = os.path.join(temp_path, f"{index}_streetview_{i}.tif")
        image.save(temp_output_path)

        if save_streetview:
            sv_output = os.path.join(sv_path, f"{index}_streetview_{i}.tif")
            image.save(sv_output)

    # list images we just wrote
    files = sorted([f for f in os.listdir(temp_path) if f.endswith(".tif")])
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
        mask_file = os.path.join(folder_path, f"{index}_streetview_0.tif")
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


    # ---------------------------------------------------
    # 6. Clean temp directory
    # ---------------------------------------------------
    delete_files(temp_path)
