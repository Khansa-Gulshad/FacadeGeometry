from modules.process_data import *
from modules.road_network import get_road_network, select_points_on_road_network, attach_road_angle

import google_streetview.api
import geopandas as gpd
import random
import math
from geojson import Feature, FeatureCollection
import os
import requests
from PIL import Image
from urllib.parse import urlparse

# Function adapted from iabrilvzqz
# GitHub: https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility-GSV
# Prepare facade folders, create road network, and create features. 
# Load pre-existing features instead of generating them
def create_features(city, access_token, distance, num_sample_images, begin, end, save_roads_points, i=0,  bbox=None):
    # Fallback to env var or default only if not provided
    print(f"Creating features for city {city}")

    # 1) Roads with road_angle
    roads = get_road_network(city, bbox=bbox)                                     # change 4
    if roads is None or (hasattr(roads, "empty") and roads.empty):
        print("No roads found. Returning empty features.")
        return gpd.GeoDataFrame()

    # 2) Sample points along roads
    pts = select_points_on_road_network(roads, N=max(1, int(distance)))           # change 5

    # 3) Attach road_angle to points (both are in projected CRS), then convert to WGS84 for GSV
    features = attach_road_angle(pts, roads, max_distance=1.0)                    # change 6
    features = features.to_crs(4326)                                              # change 7

    # 4) Add ids + sampling flags
    if 'id' not in features.columns: features['id'] = range(len(features))
    features = features.sort_values('id')
    features['save_sample'] = False

    # choose subset
    limit = (10**9 if (num_sample_images is None or (isinstance(num_sample_images,(float,int)) and math.isinf(num_sample_images)))
             else int(num_sample_images))
    k = int(min(limit, len(features)))
    if k > 0:
        pick = random.sample(range(len(features)), k)
        features.loc[features.index[pick], 'save_sample'] = True

    if begin is not None and end is not None:
        features = features.iloc[begin:end]

    # 5) Optional: save roads/points like your old flow
    if save_roads_points:
        outp = os.path.join("/mnt/project/pt01183/facade_results", city, "points"); os.makedirs(outp, exist_ok=True)
        outr = os.path.join("/mnt/project/pt01183/facade_results", city, "roads");  os.makedirs(outr, exist_ok=True)
        roads.to_file(os.path.join(outr, f"roads_{i}.gpkg"),   driver="GPKG", layer=f"roads_{i}")
        features.to_file(os.path.join(outp, f"points_{i}.gpkg"), driver="GPKG", layer=f"points_{i}")

    return features
# For each feature, calculates the facade greening potential score (GPS).
# Returns a list of GeoJSON features.

# Function to validate the URL
def is_valid_url(url):
    try:
        result = urlparse(url)
        # Check if the URL has a scheme (http/https) and netloc (domain)
        return all([result.scheme, result.netloc])
    except Exception as e:
        return False


def calculate_usable_wall_ratios(
    features,
    city,
    sam,
    access_token,
    save_streetview,
    bbox_i="0",
    radius=15,
    pitch="25"
):
    # base output root (consistent with your other paths)
    BASE_ROOT = "/mnt/project/pt01183/facade_results"

    usable_ratio = []

    # temp output dirs where segment_images writes masks
    facades_dir = os.path.join(BASE_ROOT, city, "temp_seg_facades")
    windows_dir = os.path.join(BASE_ROOT, city, "temp_seg_windows")

    # ensure folders exist (segment_images expects these)
    from modules.process_data import prepare_folders
    prepare_folders(city)

    for index, row in features.iterrows():
        # guard: only process marked samples (default True if column missing)
        if not row.get("save_sample", True):
            continue

        lat, lon = row.geometry.y, row.geometry.x

        # normalize road_angle -> [0, 360)
        try:
            ra = float(row.get("road_angle", 0.0))
            if math.isnan(ra):
                ra = 0.0
        except Exception:
            ra = 0.0
        ra = (ra % 360 + 360) % 360

        # ---------- 1) Discover pano near the point ----------
        try:
            params0 = [{
                "size": "640x640",
                "location": f"{lat},{lon}",
                "heading": "0",
                "fov": "90",
                "pitch": pitch,            # slight tilt up to reduce road
                "radius": str(radius),
                "key": access_token,
            }]
            res0 = google_streetview.api.results(params0)
            meta0 = res0.metadata[0] if getattr(res0, "metadata", None) else {}
            status = meta0.get("status")
            if status not in (None, "OK"):
                print(f"[GSV] No coverage/status={status} at idx {index}")
                continue

            pano_id = meta0.get("pano_id")
            if not pano_id:
                print(f"[GSV] Missing pano_id for idx {index}")
                continue

        except Exception as e:
            print(f"[GSV] Failed pano discovery for idx {index}: {e}")
            continue

        # ---------- 2) Fetch the two perpendicular views directly ----------
        left_h  = (ra - 90) % 360
        right_h = (ra + 90) % 360

        def fetch_view(pano_id_, heading, key, radius_, fov="90", pitch_="25"):
            import google_streetview.api, requests
            from PIL import Image
            params = [{
                "size": "640x640",
                "pano": pano_id_,
                "heading": str(heading),
                "pitch": pitch_,
                "fov": fov,
                "radius": str(radius_),
                "key": key,
            }]
            res = google_streetview.api.results(params)
            meta = res.metadata[0] if getattr(res, "metadata", None) else {}
            if meta.get("status") not in (None, "OK"):
                return None
            url = res.links[0] if getattr(res, "links", None) else None
            if not url:
                return None
            try:
                return Image.open(
                    requests.get(url, stream=True, timeout=30).raw
                ).convert("RGB")
            except Exception:
                return None

        left_im  = fetch_view(pano_id, left_h,  access_token, radius, fov="90", pitch_=pitch)
        right_im = fetch_view(pano_id, right_h, access_token, radius, fov="90", pitch_=pitch)

        # optional: crop like Mapillary (trim sky/road)
        def crop_sv(img, crop_top=0.05, crop_bottom=0.30):
            w, h = img.size
            return img.crop((0, int(h * crop_top), w, int(h * (1 - crop_bottom))))

        images = []
        for im in (left_im, right_im):
            if im is not None:
                images.append(crop_sv(im, 0.05, 0.30))

        if not images:
            print(f"[GSV] No perpendicular views for idx {index}")
            continue

        # ---------- 3) Segment the two views ----------
        segment_images(sam, images, city, index, save_streetview)

        # ---------- 4) Read masks & compute ratios ----------
        facades_segs, windows_segs = load_images(facades_dir, windows_dir)
        widths, heights, ratios = [], [], []

        if (not facades_segs and not windows_segs) or (not facades_segs):
            ratio = 0
        elif not windows_segs:
            ratio = 1
        else:
            if len(facades_segs) >= len(windows_segs):
                for name in facades_segs:
                    with Image.open(os.path.join(facades_dir, name)) as fseg:
                        fcnt = count_white_pixels(fseg)
                        w, h = fseg.size
                    wcnt = 0
                    wpath = os.path.join(windows_dir, name)
                    if os.path.isfile(wpath):
                        with Image.open(wpath) as wseg:
                            wcnt = count_white_pixels(wseg)
                    r = (fcnt - wcnt) / fcnt if fcnt > 0 else 0
                    widths.append(w); heights.append(h); ratios.append(r)
            else:
                for name in windows_segs:
                    with Image.open(os.path.join(windows_dir, name)) as wseg:
                        wcnt = count_white_pixels(wseg)
                        w, h = wseg.size
                    fcnt = 0
                    fpath = os.path.join(facades_dir, name)
                    if os.path.isfile(fpath):
                        with Image.open(fpath) as fseg:
                            fcnt = count_white_pixels(fseg)
                    r = (fcnt - wcnt) / fcnt if fcnt > 0 else 0
                    widths.append(w); heights.append(h); ratios.append(r)

        try:
            rL, rR = ratios[0], ratios[1]
            wL, wR = widths[0], widths[1]
            hL, hR = heights[0], heights[1]
            WAR = calculate_WAR(wL, hL, rL, wR, hR, rR)
        except Exception:
            rL, rR = (ratios[0], None) if ratios else (None, None)
            WAR = ratios[0] if len(ratios) == 1 else 0

        # ---------- 5) Move temp masks to final folders ----------
        suffix = str(bbox_i)
        facades_dst = os.path.join(BASE_ROOT, city, "seg_facades", suffix)
        windows_dst = os.path.join(BASE_ROOT, city, "seg_windows", suffix)
        os.makedirs(facades_dst, exist_ok=True)
        os.makedirs(windows_dst, exist_ok=True)
        move_files(facades_dir, facades_dst)
        move_files(windows_dir, windows_dst)

        # (Optional) record paths of saved streetview inputs (if you enabled save_streetview)
        image_paths = []
        if save_streetview:
            sv_dir = os.path.join(BASE_ROOT, city, "sv_images")
            # segment_images saves as f"{index}_streetview_0.tif" and "_1.tif"
            for i_side in (0, 1):
                pth = os.path.join(sv_dir, f"{index}_streetview_{i_side}.tif")
                if os.path.isfile(pth):
                    image_paths.append(pth)

        # ---------- 6) Record feature ----------
        usable_ratio.append(Feature(
            geometry=row.geometry.__geo_interface__,
            properties={
                "ratio_left": rL,
                "ratio_right": rR,
                "GPS": round(WAR, 2),
                "image_paths": image_paths,   # list if saved, else []
                "pano_id": pano_id,
                "road_angle": ra,
                "idx": int(index),
            },
        ))

    return usable_ratio

# Saves usable facade ratios (GPS) as a geopackage file
def save_usable_wall_ratios(city, usable_ratios):
    # Save points and ratios as GeoJSON FeatureCollection
    features_file = f'{city}_features.gpkg'
    features_path = os.path.join("/mnt/project/pt01183/facade_results", city)
    feature_collection = FeatureCollection(usable_ratios)

    # Convert it to a GeoDataFrame from the features
    gdf = gpd.GeoDataFrame.from_features(feature_collection["features"])

    # Ensure the 'geometry' column is set
    if 'geometry' not in gdf.columns:
        print("[ERROR] No geometry column found in the GeoDataFrame.")
        return

    # Set the 'geometry' column as the active geometry column if not already set
    gdf.set_geometry('geometry', inplace=True)

    # Ensure CRS is set correctly (WGS 84 - EPSG:4326)
    gdf.set_crs('EPSG:4326', inplace=True)

    # Save the GeoDataFrame to a GeoPackage
    gdf.to_file(f'{features_path}/{features_file}', driver="GPKG")

    print(f"Saved features to {features_file}")kage
  gdf = gpd.GeoDataFrame.from_features(feature_collection["features"])
  gdf.set_crs('EPSG:4326', inplace=True)
  gdf.to_file(f'{features_path}/{features_file}', driver="GPKG")
