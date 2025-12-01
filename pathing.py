import heapq
import plotly.graph_objects as go

###############################################################################
# Drawing functions
###############################################################################

def draw_way(fig, nodes: dict, way: dict):
    """Using plotly graph to draw the way passed into the function

    Args:
        fig (Figure instance): Plotly Figure instance
        nodes (dict): Dictionary of nodes {'node_id': {'lat':..., 'lon':..., 'label':...}}
        way (dict): Dictionary representing a way to be drawn {'id':..., 'from':..., 'to':...,
            'name':..., 'type':(primary, secondary, tertiary, service), 'base_time':..., 
            'accident_severity':(0-4), 'final_time':...}
    """
    way_color = "black"
    if way['type'] == "primary":
        way_color = "deepskyblue"
    elif way['type'] == "secondary":
        way_color = "yellow"
    elif way['type'] == "tertiary":
        way_color = "orange"
    elif way['type'] == "service":
        way_color = "slategray"
    
    hovertext = (f"{way['id']}: {way['name']}<br>"
                 f"From: {way['from']}<br>"
                 f"To: {way['to']}<br>")
    
    fig.add_trace(go.Scattermap(
        lat=[nodes[way['from']]['lat'], nodes[way['to']]['lat']],
        lon=[nodes[way['from']]['lon'], nodes[way['to']]['lon']],
        mode='lines',
        name=way['name'],
        line=dict(
            width=3,
            color=way_color
        ),
        text=hovertext,
        showlegend=False
    ))

    # Draw invisible trace marker along the path to display hovertext information along the line
    # Interpolate an array of nodes along the way for better hover effect
    density = 10
    lat_points = []
    lon_points = []
    for i in range(density + 1):
        frac = i / density
        lat_points.append(nodes[way['from']]['lat'] * (1 - frac) + nodes[way['to']]['lat'] * frac)
        lon_points.append(nodes[way['from']]['lon'] * (1 - frac) + nodes[way['to']]['lon'] * frac)
    
    fig.add_trace(go.Scattermap(
        lat=lat_points,
        lon=lon_points,
        mode='markers',
        marker=dict(
            size=2,
            color='rgba(0,0,0,0)'  # invisible markers
        ),
        hoverinfo='text',
        text=hovertext,
        showlegend=False
    ))


def draw_path(fig, nodes: dict, ways: list, path: list):
    """Using plotly graph to draw the path passed into the function

    Args:
        fig (Figure instance): Plotly Figure instance
        nodes (dict): Dictionary of nodes {'node_id': {'lat':..., 'lon':..., 'label':...}}
        ways (list[dict]): List of ways given as dictionaries
        path (list): List of node ids representing the path to be drawn
    """
    if not path or len(path) < 2:
        return
    
    # Draw the path as a thick line on top of the map
    lat_points = [nodes[node_id]['lat'] for node_id in path]
    lon_points = [nodes[node_id]['lon'] for node_id in path]
    
    fig.add_trace(go.Scattermap(
        lat=lat_points,
        lon=lon_points,
        mode='lines+markers',
        line=dict(
            width=5,
            color='red'
        ),
        marker=dict(
            size=8,
            color='red'
        ),
        name='Path',
        hoverinfo='text',
        text=[f"Node: {nid}" for nid in path],
        showlegend=True
    ))


###############################################################################
# K-nearest snapping
###############################################################################

def k_nearest_graph_nodes(lat, lon, osm_nodes, graph, k=5):
    """Find the k nearest nodes in the graph to a given lat/lon position
    
    Args:
        lat (float): Latitude of the target position
        lon (float): Longitude of the target position
        osm_nodes (dict): Dictionary mapping node IDs to (lat, lon) tuples
        graph (dict): Graph adjacency list (to verify nodes exist in graph)
        k (int): Number of nearest neighbors to return
    
    Returns:
        list: List of up to k nearest node IDs
    """
    candidates = []
    for nid, (nlat, nlon) in osm_nodes.items():
        if nid not in graph:
            continue
        d_lat = lat - nlat
        d_lon = lon - nlon
        d2 = d_lat * d_lat + d_lon * d_lon
        candidates.append((d2, nid))
    
    candidates.sort(key=lambda x: x[0])
    return [nid for d2, nid in candidates[:k]]


###############################################################################
# Dijkstra in road graph
###############################################################################

def dijkstra_path(graph, start_id, goal_id):
    """Find shortest path from start to goal using Dijkstra's algorithm
    
    Args:
        graph (dict): Adjacency list where graph[node] = [(neighbor, weight), ...]
        start_id: Starting node ID
        goal_id: Goal node ID
    
    Returns:
        list: Path from start to goal as list of node IDs, or None if no path exists
    """
    dist = {start_id: 0.0}
    prev = {}
    pq = [(0.0, start_id)]
    visited = set()

    while pq:
        cur_d, cur = heapq.heappop(pq)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal_id:
            break

        for nbr, w in graph.get(cur, []):
            nd = cur_d + w
            if nbr not in dist or nd < dist[nbr]:
                dist[nbr] = nd
                prev[nbr] = cur
                heapq.heappush(pq, (nd, nbr))

    if goal_id not in dist:
        return None

    path = []
    node = goal_id
    while True:
        path.append(node)
        if node == start_id:
            break
        node = prev.get(node)
        if node is None:
            return None
    path.reverse()
    return path


def path_length_km(path, graph):
    """Calculate total length of a path in kilometers
    
    Args:
        path (list): List of node IDs representing the path
        graph (dict): Adjacency list where graph[node] = [(neighbor, weight), ...]
    
    Returns:
        float: Total path length in kilometers
    """
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        for nbr, w in graph.get(a, []):
            if nbr == b:
                total += w
                break
    return total


###############################################################################
# Helper functions
###############################################################################

def color_for_highway(hwy_type):
    """Return color for a given highway type
    
    Args:
        hwy_type (str): Highway type (primary, secondary, tertiary, service, etc.)
    
    Returns:
        str: Color name for the highway type
    """
    h = (hwy_type or "").lower()
    if h == "primary":
        return "deepskyblue"
    elif h == "secondary":
        return "purple"
    elif h == "tertiary":
        return "darkblue"
    elif h == "service":
        return "slategray"
    else:
        return "orange"  # fallback color


def get_map_center(nodes):
    """Calculate center point of all nodes
    
    Args:
        nodes (dict): Dictionary of nodes with 'lat' and 'lon' keys
    
    Returns:
        tuple: (center_lat, center_lon)
    """
    lats = [info["lat"] for info in nodes.values()]
    lons = [info["lon"] for info in nodes.values()]
    return sum(lats) / len(lats), sum(lons) / len(lons)


###############################################################################
# Graph building from OSM data
###############################################################################

def build_road_graph(osm_nodes, osm_ways):
    """Build a graph from OSM nodes and ways for pathfinding
    
    Args:
        osm_nodes (dict): Dictionary mapping node IDs to (lat, lon) tuples
        osm_ways (list): List of way dictionaries with 'from_osm' and 'to_osm' keys
    
    Returns:
        dict: Adjacency list where graph[node_id] = [(neighbor_id, distance_km), ...]
    """
    graph = {}
    
    for way in osm_ways:
        from_id = way['from_osm']
        to_id = way['to_osm']
        
        if from_id not in osm_nodes or to_id not in osm_nodes:
            continue
        
        # Calculate distance using Haversine formula
        lat1, lon1 = osm_nodes[from_id]
        lat2, lon2 = osm_nodes[to_id]
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        
        # Add bidirectional edges (assuming roads are bidirectional)
        if from_id not in graph:
            graph[from_id] = []
        if to_id not in graph:
            graph[to_id] = []
        
        graph[from_id].append((to_id, distance))
        graph[to_id].append((from_id, distance))
    
    return graph


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
    
    Returns:
        float: Distance in kilometers
    """
    import math
    
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c