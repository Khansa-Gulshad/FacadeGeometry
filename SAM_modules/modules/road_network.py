import geopandas as gpd
import osmnx as ox
import math
from shapely.geometry import Point
import networkx as nx

# Function adapted from iabrilvzqz
# GitHub: https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility-GSV
# Generates a road network from either a placename or bounding box using OpenStreetMap data
# Also saves the road angle of each road segment
def get_road_network(city, bbox):
    # Use a custom filter to only get car-driveable roads that are not motorways or trunk roads
    cf = '["highway"~"primary|secondary|tertiary|residential|primary_link|secondary_link|tertiary_link|living_street|service|unclassified"]'

    try:
      if bbox:
        G = ox.graph_from_bbox(bbox[3], bbox[1], bbox[2], bbox[0], simplify=True, custom_filter=cf)
      else:
        G = ox.graph_from_place(city, simplify=True, custom_filter=cf)
    except:
      # If there are no roads, return an empty gdf
      return gpd.GeoDataFrame()

    # Create a set to store unique road identifiers
    unique_roads = set()

    # Create a new graph to store the simplified road network
    G_simplified = G.copy()

    # Iterate over each road segment
    for u, v, key in G.edges(keys=True):
      # Check if the road segment is a duplicate
      if (v, u) in unique_roads:
        # Remove the duplicate road segment
        G_simplified.remove_edge(u, v, key)
      else:
        # Add the road segment to the set of unique roads
        unique_roads.add((u, v))

        y0, x0 = G.nodes[u]['y'], G.nodes[u]['x']
        y1, x1 = G.nodes[v]['y'], G.nodes[v]['x']

        # Calculate the angle from North (in radians)
        angle_from_north = math.atan2(x1 - x0, y1 - y0)

        # Convert the angle to degrees
        angle_from_north_degrees = math.degrees(angle_from_north)

        if angle_from_north_degrees < 0:
          angle_from_north_degrees += 360.0
        
        # Add the road angle as a new attribute to the edge
        G_simplified.edges[u,v,key]['road_angle'] = angle_from_north_degrees

    # Update the graph with the simplified road network
    G = G_simplified
    
    # Project the graph from latitude-longitude coordinates to a local projection (in meters)
    G_proj = ox.project_graph(G)

    # Convert the projected graph to a GeoDataFrame
    # This function projects the graph to the UTM CRS for the UTM zone in which the graph's centroid lies
    _ , edges = ox.graph_to_gdfs(G_proj) 

    return edges, G_proj

def remove_intersection_points(points, G, min_dist=12.0):
    """
    Remove points that lie close to road intersections.
    """
    # get intersection nodes (degree >= 3)
    intersections = [
        n for n, deg in dict(G.degree()).items() if deg >= 3
    ]

    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
    inter_nodes = nodes.loc[intersections]

    points_m = points.to_crs(nodes.crs)
    inter_nodes = inter_nodes.to_crs(nodes.crs)

    mask = []
    for pt in points_m.geometry:
        d = inter_nodes.distance(pt).min()
        mask.append(d > min_dist)

    return points.loc[mask].reset_index(drop=True)

# Function courtesy of iabrilvzqz
# GitHub: https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility-GSV
# Get a gdf of points over a road network with a N distance between them
def select_points_on_road_network(roads, N=50):
  points = []
  
  # Iterate over each road
  for row in roads.itertuples(index=True, name='Road'):
    # Get the LineString object from the geometry
    linestring = row.geometry
    index = row.Index

    # Calculate the distance along the linestring and create points every 50 meters
    for distance in range(0, int(linestring.length), N):
      # Get the point on the road at the current position
      point = linestring.interpolate(distance)

      # Add the curent point to the list of points
      points.append([point, index])
  
  # Convert the list of points to a GeoDataFrame
  gdf_points = gpd.GeoDataFrame(points, columns=["geometry", "road_index"], geometry="geometry")

  # Set the same CRS as the road dataframes for the points dataframe
  gdf_points.set_crs(roads.crs, inplace=True)

  # Drop duplicate rows based on the geometry column
  gdf_points = gdf_points.drop_duplicates(subset=['geometry'])
  gdf_points = gdf_points.reset_index(drop=True)

  return gdf_points


# Function courtesy of iabrilvzqz
# GitHub: https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility-GSV
# This function extracts the features for a given tile

# Function adapted from iabrilvzqz
# GitHub: https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility-GSV


  # Transform the coordinate reference system to EPSG 4326
  points.to_crs(epsg=4326, inplace=True)

  return points
