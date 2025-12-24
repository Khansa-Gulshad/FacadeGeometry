import geopandas as gpd
import numpy as np
import osmnx as ox
import math
from shapely.geometry import Point
from shapely.ops import orient


# -------------------------------------------------------
# LOAD BUILDINGS FROM OSM (city name OR bbox)
# -------------------------------------------------------
def snap_points_to_roads(
    gdf_points,
    city=None,
    bbox=None,
    min_dist_to_intersection=12.0
):
    """
    Snap façade sampling points to nearest drivable road,
    excluding points near intersections.

    Args:
        gdf_points : GeoDataFrame (EPSG:4326)
        city       : city name (str) OR
        bbox       : [minx, miny, maxx, maxy] in lon/lat
        min_dist_to_intersection : meters

    Returns:
        GeoDataFrame with snapped points
    """

    print("\n========== SNAPPING POINTS TO ROADS ==========")

    # ------------------------------------------------------------------
    # 1. Load drivable road network
    # ------------------------------------------------------------------
    if bbox is not None:
        G = ox.graph_from_bbox(
            north=bbox[3], south=bbox[1],
            east=bbox[2], west=bbox[0],
            network_type="drive"
        )
    else:
        G = ox.graph_from_place(city, network_type="drive")

    # Project graph to meters
    G = ox.project_graph(G)

    # Nodes & edges
    nodes, edges = ox.graph_to_gdfs(G)

    # ------------------------------------------------------------------
    # 2. Project points to same CRS
    # ------------------------------------------------------------------
    gdf = gdf_points.to_crs(edges.crs)

    snapped_pts = []

    for idx, row in gdf.iterrows():
        pt = row.geometry

        # nearest edge
        try:
            u, v, key = ox.distance.nearest_edges(G, pt.x, pt.y)
        except Exception:
            continue

        edge = edges.loc[(u, v, key)]

        # ------------------------------------------------------------------
        # 3. Reject points near intersections
        # ------------------------------------------------------------------
        u_pt = nodes.loc[u].geometry
        v_pt = nodes.loc[v].geometry

        if pt.distance(u_pt) < min_dist_to_intersection:
            continue
        if pt.distance(v_pt) < min_dist_to_intersection:
            continue

        # ------------------------------------------------------------------
        # 4. Snap to edge geometry
        # ------------------------------------------------------------------
        line = edge.geometry
        snapped_geom = line.interpolate(line.project(pt))

        new_row = row.copy()
        new_row.geometry = snapped_geom
        snapped_pts.append(new_row)

    snapped_gdf = gpd.GeoDataFrame(snapped_pts, crs=edges.crs)

    # Back to WGS84 for GSV
    snapped_gdf = snapped_gdf.to_crs(4326)

    print(f"✓ Snapped {len(snapped_gdf)} points to roads (no intersections)")
    return snapped_gdf

def get_buildings(city=None, bbox=None):
    """
    Download building footprints directly from OpenStreetMap.

    Args:
        city (str): e.g., "Gdańsk, Poland"
        bbox (list): [minx, miny, maxx, maxy] (lon/lat)
    
    Returns:
        GeoDataFrame of building polygons (projected to meters)
    """

    print("\n========== LOADING BUILDINGS ==========")

    # OSM building filter
    custom_filter = '["building"~"."]'

    try:
        if bbox is not None:
            print(f"Downloading buildings from BBOX: {bbox}")
            buildings = ox.geometries_from_bbox(
                north=bbox[3],
                south=bbox[1],
                east=bbox[2],
                west=bbox[0],
                tags={"building": True}
            )
        else:
            print(f"Downloading buildings for: {city}")
            buildings = ox.geometries_from_place(
                city,
                tags={"building": True}
            )
    except Exception as e:
        print(f"Error loading buildings: {e}")
        return gpd.GeoDataFrame()

    # Keep only polygons
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]

    print(f"✓ Found {len(buildings)} buildings")

    # Reproject to local CRS in meters
    buildings = buildings.to_crs(3857)

    return buildings



# -------------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------------

def compute_bearing(x0, y0, x1, y1):
    """Returns bearing of wall (in degrees, 0°=North)."""
    ang = math.degrees(math.atan2(x1 - x0, y1 - y0))
    if ang < 0:
        ang += 360
    return ang


def outward_normal(bearing):
    """
    For a wall with bearing θ,
    outward normal direction = θ - 90° (converted to 0–360)
    """
    return (bearing - 90) % 360


def move_point(px, py, bearing_deg, dist_m):
    """Move (px,py) outward from façade."""
    rad = math.radians(bearing_deg)
    return (
        px + dist_m * math.sin(rad),
        py + dist_m * math.cos(rad)
    )



# -------------------------------------------------------
# MAIN FUNCTION: GENERATE FACADE SAMPLING POINTS
# -------------------------------------------------------

def get_facade_sampling_points(buildings, offset_m=8.0):
    """
    Convert building polygons to façade sampling points.

    Args:
        buildings: GDF from get_buildings()
        offset_m: distance to offset point outward from façade
    
    Returns:
        GeoDataFrame with:
            - geometry (sampling point)
            - building_id
            - segment_id
            - facade_heading (camera heading)
    """

    print("\n========== GENERATING FACADE SAMPLING POINTS ==========")

    buildings = buildings.copy()
    buildings = buildings.explode(index_parts=False)

    points = []

    for bid, row in buildings.iterrows():
        poly = orient(row.geometry, sign=1.0)  # ensure clockwise
        if poly.geom_type != "Polygon":
            continue

        coords = list(poly.exterior.coords)

        for i in range(len(coords) - 1):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]

            # compute wall length
            L = math.dist((x0, y0), (x1, y1))
            if L < 4:   # skip tiny noisy edges
                continue

            # midpoint
            mx = (x0 + x1) / 2
            my = (y0 + y1) / 2

            wall_bearing = compute_bearing(x0, y0, x1, y1)

            # candidate normal
            n1 = (wall_bearing - 90) % 360
            n2 = (wall_bearing + 90) % 360

            # test which normal points away from centroid
            cx, cy = poly.centroid.coords[0]

            tx1, ty1 = move_point(mx, my, n1, 1.0)
            tx2, ty2 = move_point(mx, my, n2, 1.0)

            d1 = math.dist((tx1, ty1), (cx, cy))
            d2 = math.dist((tx2, ty2), (cx, cy))

            normal = n1 if d1 > d2 else n2
            # move outward
            sx, sy = move_point(mx, my, normal, offset_m)

            # camera faces the façade → look *opposite* the outward normal
            camera_heading = (normal + 180) % 360

            points.append({
                "geometry": Point(sx, sy),
                "building_id": bid,
                "segment_id": i,
                "facade_heading": camera_heading,
                "wall_length": L,
            })

    gdf = gpd.GeoDataFrame(points, geometry="geometry", crs=buildings.crs)

    # Return in lat/lon for GSV API
    gdf = gdf.to_crs(4326)

    print(f"✓ Created {len(gdf)} façade sampling points")
    return gdf
