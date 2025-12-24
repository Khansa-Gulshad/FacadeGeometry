from SAM_modules.modules.process_data import *
from SAM_modules.modules.road_network import (get_road_network, select_points_on_road_network, remove_intersection_points, attach_road_angle)


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
def create_features(
    city,
    N=20,
    bbox=None,
    district=None,
    save_roads_points=True
):
    print(f"Creating road-based sampling points for {city}")

    # 1) Road network
    roads, G = get_road_network(city, bbox)
    if roads.empty or G is None:
        raise RuntimeError("No roads extracted")

    # 2) Sample points
    points = select_points_on_road_network(roads, N=N)

    # 3) Remove intersections
    points = remove_intersection_points(points, G, min_dist=12.0)

    # 4) Attach road angle
    points = attach_road_angle(points, roads)

    # 5) Clip to district (IMPORTANT)
    if district is not None:
        points = gpd.clip(points, district)

    # 6) Save
    if save_roads_points:
        outp = os.path.join(
            "/mnt/project/pt01183/facade_results",
            city,
            "points"
        )
        os.makedirs(outp, exist_ok=True)
        points.to_file(
            os.path.join(outp, f"road_points_N{N}.gpkg"),
            driver="GPKG"
        )

    return points

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
    BASE_ROOT = "/mnt/project/pt01183/facade_results"
    city_root = os.path.join(BASE_ROOT, city)

    usable_ratio = []

    # ensure folders exist
    from modules.process_data import prepare_folders
    prepare_folders(city)

    seg_npz_dir = os.path.join(city_root, "seg_npz")

    # ---------- helper: fetch one view aligned with facade ----------

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

    # ---------- MAIN LOOP ----------
    for index, row in features.iterrows():
        if not row.get("save_sample", True):
            continue

        lat, lon = row.geometry.y, row.geometry.x

        # heading towards façade
        try:
            ra = float(row.get("road_angle", 0.0))
            if math.isnan(ra):
                ra = 0.0
        except Exception:
            ra = 0.0
        ra = (ra % 360 + 360) % 360

        # 1) discover pano
        try:
            params0 = [{
                "size": "640x640",
                "location": f"{lat},{lon}",
                "heading": "0",
                "fov": "90",
                "pitch": pitch,
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

        # 2) fetch façade view
        headings = [
        (ra - 90) % 360,   # left façade
        (ra + 90) % 360    # right façade
        ]

        images = []
        for h in headings:
            img = fetch_view(pano_id, h, access_token, radius, fov="90", pitch_=pitch)
            if img is not None:
                images.append(img)

        # optional: crop SVI
        def crop_sv(img, crop_top=0.05, crop_bottom=0.30):
            w, h = img.size
            return img.crop((0, int(h * crop_top), w, int(h * (1 - crop_bottom))))

        im_c = crop_sv(im, 0.05, 0.30)

        # 3) segment (this will create seg_npz/{index}.npz & seg_vis/{index}.png)
        segment_images(sam, [im_c], city, index, save_streetview)

        # 4) load label map and compute usable façade ratio
        npz_path = os.path.join(seg_npz_dir, f"{index}.npz")
        if not os.path.exists(npz_path):
            print(f"[SAM] Missing NPZ for idx {index}")
            continue

        seg = np.load(npz_path)["seg"]

        # labels: 1=ground, 2=facade, 3=windows/doors, 4=sky
        fcnt = (seg == 2).sum()
        wcnt = (seg == 3).sum()
        ratio = (fcnt - wcnt) / fcnt if fcnt > 0 else 0.0

        # For compatibility, treat this as WAR (one view only)
        WAR = ratio
        rL, rR = ratio, None

        # 5) optionally attach SVI path
        image_paths = []
        if save_streetview:
            sv_dir = os.path.join(city_root, "sv_images")
            pth = os.path.join(sv_dir, f"{index}_streetview_0.tif")
            if os.path.isfile(pth):
                image_paths.append(pth)

        # 6) record feature
        usable_ratio.append(Feature(
            geometry=row.geometry.__geo_interface__,
            properties={
                "ratio_left": rL,
                "ratio_right": rR,
                "GPS": round(WAR, 2),
                "image_paths": image_paths,
                "pano_id": pano_id,
                "road_angle": ra,
                "idx": int(index),
            },
        ))

    return usable_ratio


# Saves usable facade ratios (GPS) as a geopackage file
def save_usable_wall_ratios(city, usable_ratios):
    features_file = f"{city}_features.gpkg"
    features_path = os.path.join("/mnt/project/pt01183/facade_results", city)
    feature_collection = FeatureCollection(usable_ratios)

    gdf = gpd.GeoDataFrame.from_features(feature_collection["features"])

    if "geometry" not in gdf.columns:
        print("[ERROR] No geometry column found in the GeoDataFrame.")
        return

    gdf.set_geometry("geometry", inplace=True)
    gdf.set_crs("EPSG:4326", inplace=True)

    gdf.to_file(f"{features_path}/{features_file}", driver="GPKG")
    print(f"Saved features to {features_file}")
