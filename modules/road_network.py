import geopandas as gpd
import osmnx as ox
import math
from shapely.geometry import Point
import networkx as nx


# -------------------------------------------------------
# ROAD NETWORK EXTRACTION
# -------------------------------------------------------
def get_road_network(city, bbox):
    """
    Extract drivable road network from OSM and compute road angles.

    Returns:
        edges (GeoDataFrame, meters)
        G_proj (networkx graph, meters)
    """

    cf = '["highway"~"primary|secondary|tertiary|residential|primary_link|secondary_link|tertiary_link|living_street|service|unclassified"]'

    try:
        if bbox:
            # bbox = [minx, miny, maxx, maxy]
            north = bbox[3]
            south = bbox[1]
            east  = bbox[2]
            west  = bbox[0]

            G = ox.graph_from_bbox(north, south, east, west, simplify=True, custom_filter=cf)
        else:
            G = ox.graph_from_place(city, simplify=True, custom_filter=cf)
    except Exception as e:
        print(f"[OSM] Road extraction failed: {e}")
        return gpd.GeoDataFrame(), None

    # Remove duplicate reversed edges + compute road angle
    unique_roads = set()
    G_simplified = G.copy()

    for u, v, key in G.edges(keys=True):
        if (v, u) in unique_roads:
            G_simplified.remove_edge(u, v, key)
            continue

        unique_roads.add((u, v))

        y0, x0 = G.nodes[u]["y"], G.nodes[u]["x"]
        y1, x1 = G.nodes[v]["y"], G.nodes[v]["x"]

        angle = math.degrees(math.atan2(x1 - x0, y1 - y0))
        if angle < 0:
            angle += 360.0

        G_simplified.edges[u, v, key]["road_angle"] = angle

    # Project to meters
    G_proj = ox.project_graph(G_simplified)

    _, edges = ox.graph_to_gdfs(G_proj)

    return edges, G_proj


# -------------------------------------------------------
# REMOVE INTERSECTION POINTS
# -------------------------------------------------------
def remove_intersection_points(points, G, min_dist=12.0):
    """
    Remove points closer than min_dist (meters) to intersections.
    """

    intersections = [n for n, deg in dict(G.degree()).items() if deg >= 3]

    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    inter_nodes = nodes.loc[intersections]

    points_m = points.to_crs(nodes.crs)

    mask = [
        inter_nodes.distance(pt).min() > min_dist
        for pt in points_m.geometry
    ]

    return points.loc[mask].reset_index(drop=True)


# -------------------------------------------------------
# SAMPLE POINTS ALONG ROADS
# -------------------------------------------------------
def select_points_on_road_network(roads, N=50):
    """
    Sample points every N meters along road centerlines.
    """

    pts = []

    for row in roads.itertuples(index=True):
        line = row.geometry
        rid = row.Index

        N = max(1, int(N))
        for d in range(0, int(line.length), N):
            pts.append([line.interpolate(d), rid])

    gdf = gpd.GeoDataFrame(
        pts,
        columns=["geometry", "road_index"],
        geometry="geometry",
        crs=roads.crs
    )

    gdf = gdf.drop_duplicates(subset=["geometry"]).reset_index(drop=True)

    return gdf
    
def attach_road_angle(points: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, max_distance=1.0) -> gpd.GeoDataFrame:
    """
    Nearest-edge spatial join to bring 'road_angle' from road edges onto points.
    Assumes BOTH 'points' and 'roads' are in the SAME (projected) CRS (what osmnx returns).
    """
    out = gpd.sjoin_nearest(points, roads[['geometry', 'road_angle']], how='left', max_distance=max_distance)
    # clean up join artifacts
    drop_cols = [c for c in out.columns if c.startswith('index_right')]
    out = out.drop(columns=drop_cols, errors='ignore')
    return out
