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
        mode='lines' ,
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

def k_nearest_graph_nodes(lat, lon, osm_nodes, graph, k=1):
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

################################################################################
# find_best_path_between_nodes
################################################################################
def find_best_path_between_nodes(from_node, to_node, snap_candidates, road_graph):
    """Find the best road path between two assignment nodes
    
    Returns:
        tuple: (osm_path, path_length, is_straight_line)
    """
    try:
        from_snaps = set(snap_candidates[from_node])
        to_snaps = set(snap_candidates[to_node])
        
        # Check if snap candidates overlap (nodes very close together)
        if from_snaps & to_snaps:
            print(f"  Nodes {from_node} → {to_node}: OVERLAP - using straight line")
            return None, 0, True
        
        # Try to find best path among candidate combinations
        best_path = None
        best_length = float('inf')
        
        for from_osm in snap_candidates[from_node][:3]:  # Try top 3
            for to_osm in snap_candidates[to_node][:3]:
                path = dijkstra_path(road_graph, from_osm, to_osm)
                if path and len(path) >= 2:
                    length = path_length_km(path, road_graph)
                    if length < best_length:
                        best_length = length
                        best_path = path
        
        if best_path and len(best_path) >= 2:
            print(f"  Nodes {from_node} → {to_node}: ROUTED via {len(best_path)} OSM nodes, {best_length:.3f} km")
            return best_path, best_length, False
        else:
            print(f"  Nodes {from_node} → {to_node}: NO PATH - using straight line")
            return None, 0, True
    except Exception as e:
        print(f"  ERROR in find_best_path_between_nodes({from_node}, {to_node}): {e}")
        import traceback
        traceback.print_exc()
        return None, 0, True

def draw_osm_roads(fig, osm_ways, osm_nodes, max_roads=50):
    """Draw OSM road network as background
    
    Args:
        fig: Plotly figure object
        osm_ways: List of OSM way dictionaries
        osm_nodes: Dictionary of OSM nodes
        max_roads: Maximum number of roads to draw (default 50)
    """
    osm_color_map = {
        "primary": "#87CEEB",
        "secondary": "#FFE4B5", 
        "tertiary": "#FFB6C1",
        "service": "#D3D3D3",
        "residential": "#E0E0E0",
        "unclassified": "#DCDCDC"
    }
    
    print(f"\nDrawing OSM roads (max {max_roads})...")
    
    for idx, osm_way in enumerate(osm_ways):
        if idx >= max_roads:
            break
        
        lat1, lon1 = osm_nodes[osm_way['from_osm']]
        lat2, lon2 = osm_nodes[osm_way['to_osm']]
        
        highway_type = osm_way['highway_type'].lower()
        color = osm_color_map.get(highway_type, "#DCDCDC")
        
        fig.add_trace(go.Scattermap(
            lat=[lat1, lat2],
            lon=[lon1, lon2],
            mode='lines',
            line=dict(width=1, color=color),
            hovertext=f"{osm_way['name']}<br>Type: {osm_way['highway_type']}",
            hoverinfo='text',
            showlegend=False,
            opacity=0.4
        ))
    
    print(f"Drew {min(len(osm_ways), max_roads)} OSM roads")

def draw_snap_connections(fig, nodes, snap_candidates, osm_nodes):
    """Draw dotted lines from assignment nodes to their nearest snap points
    
    Args:
        fig: Plotly figure object
        nodes: Dictionary of assignment nodes
        snap_candidates: Dictionary mapping node IDs to their snap candidates
        osm_nodes: Dictionary of OSM nodes
    """
    print(f"\nDrawing snap connections...")
    
    for node_id, node_info in nodes.items():
        if node_id in snap_candidates and snap_candidates[node_id]:
            nearest_snap = snap_candidates[node_id][0]  # Get nearest snap point
            snap_lat, snap_lon = osm_nodes[nearest_snap]
            
            # Create dotted effect by drawing multiple small segments
            num_dots = 10
            lats_dotted = []
            lons_dotted = []
            
            for j in range(num_dots):
                t = j / (num_dots - 1)
                lat_interp = node_info['lat'] + t * (snap_lat - node_info['lat'])
                lon_interp = node_info['lon'] + t * (snap_lon - node_info['lon'])
                lats_dotted.append(lat_interp)
                lons_dotted.append(lon_interp)
                if j < num_dots - 1:
                    lats_dotted.append(None)  # Break creates gap
                    lons_dotted.append(None)
            
            # Draw dotted line effect
            fig.add_trace(go.Scattermap(
                lat=lats_dotted,
                lon=lons_dotted,
                mode='lines+markers',
                line=dict(
                    width=1,
                    color='black'
                ),
                marker=dict(size=4.5, color='gray'),
                hovertext=f"Snap connection for {node_id}<br>To OSM node {nearest_snap}",
                hoverinfo='text',
                showlegend=False,
                opacity=1
            ))
            
            # Draw small marker at snap point
            fig.add_trace(go.Scattermap(
                lat=[snap_lat],
                lon=[snap_lon],
                mode='markers',
                marker=dict(size=6, color='purple', symbol='circle'),
                hovertext=f"Snap point for {node_id}<br>OSM node: {nearest_snap}",
                hoverinfo='text',
                showlegend=False
            ))
    
    print(f"Drew snap connections for {len(nodes)} nodes")

def draw_assignment_ways(fig, ways, nodes, osm_nodes, snap_candidates, road_graph, show=True):
    """Draw assignment ways on the map
    
    Args:
        fig: Plotly figure object
        ways: List of way dictionaries
        nodes: Dictionary of assignment nodes
        osm_nodes: Dictionary of OSM nodes
        snap_candidates: Dictionary mapping node IDs to snap candidates
        road_graph: Road graph for pathfinding
        show: Boolean, if False skip drawing (default True)
    """
    if not show:
        print(f"\nSkipping assignment ways")
        return
    
    prev_from = None
    prev_to = None
    
    print(f"\nDrawing {len(ways)} assignment ways...")
    
    for way_idx, way in enumerate(ways):
        try:
            if way['from'] == prev_to and way['to'] == prev_from:
                continue
            
            from_id = way['from']
            to_id = way['to']
            
            print(f"\nWay {way_idx}: {from_id} → {to_id}")
            
            osm_path, path_len, is_straight = find_best_path_between_nodes(
                from_id, to_id, snap_candidates, road_graph
            )
            
            way_color = "black"
            if way['type'] == "primary":
                way_color = "deepskyblue"
            elif way['type'] == "secondary":
                way_color = "yellow"
            elif way['type'] == "tertiary":
                way_color = "orange"
            elif way['type'] == "service":
                way_color = "slategray"
            
            if is_straight or osm_path is None:
                segment_lats = [nodes[from_id]['lat'], nodes[to_id]['lat']]
                segment_lons = [nodes[from_id]['lon'], nodes[to_id]['lon']]
                hovertext = (f"{way['id']}: {way['name']}<br>"
                            f"From: {from_id}<br>To: {to_id}<br>"
                            f"(Straight line - nodes too close)")
            else:
                segment_lats = [osm_nodes[nid][0] for nid in osm_path]
                segment_lons = [osm_nodes[nid][1] for nid in osm_path]
                hovertext = (f"{way['id']}: {way['name']}<br>"
                            f"From: {from_id}<br>To: {to_id}<br>"
                            f"Road distance: {path_len:.3f} km<br>"
                            f"OSM nodes: {len(osm_path)}")
            
            fig.add_trace(go.Scattermap(
                lat=segment_lats,
                lon=segment_lons,
                mode='lines',
                line=dict(width=4 if not is_straight else 3, color=way_color),
                hovertext=hovertext,
                hoverinfo='text',
                showlegend=False
            ))
            
            prev_from = way['from']
            prev_to = way['to']
            
        except Exception as e:
            print(f"\n!!! ERROR drawing way {way_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Drew {len(ways)} assignment ways")

path = {"nodes": [], "time": 0}
paths = []

def separate_paths(paths):
    """Separate paths into main path (fastest) and alternate paths
    
    Args:
        paths (list): List of path dictionaries sorted by time (lowest to highest)
                     Each path: {"nodes": [...], "time": ...}
    
    Returns:
        tuple: (main_path, alternate_paths)
               main_path: dict with fastest path
               alternate_paths: list of remaining paths
    """
    if not paths:
        return None, []
    
    # First path is the fastest (already sorted)
    main_path = paths[0]
    
    # Rest are alternates
    alternate_paths = paths[1:] if len(paths) > 1 else []
    
    return main_path, alternate_paths

def draw_paths(fig, paths_list, nodes, osm_nodes, snap_candidates, road_graph):
    """Draw main and alternate paths on the map
    
    Args:
        fig: Plotly figure object
        paths_list: List of path dicts [{"nodes": [...], "time": ...}] sorted by time
        nodes: Dictionary of assignment nodes
        osm_nodes: Dictionary of OSM nodes
        snap_candidates: Dictionary mapping node IDs to snap candidates
        road_graph: Road graph for pathfinding
    """
    if not paths_list or len(paths_list) == 0:
        return
    
    print(f"\nPlotting {len(paths_list)} paths...")
    
    # Separate main path and alternates
    main_path, alternate_paths = separate_paths(paths_list)
    
    # Draw alternate paths first (so main path appears on top)
    for idx, alt_path in enumerate(alternate_paths):
        route_ids = alt_path['nodes']
        print(f"\nAlternate path {idx + 1}: {route_ids} (time: {alt_path['time']})")
        
        if len(route_ids) >= 2:
            all_lats = []
            all_lons = []
            
            for i in range(len(route_ids) - 1):
                from_id = route_ids[i]
                to_id = route_ids[i + 1]
                
                osm_path, path_len, is_straight = find_best_path_between_nodes(
                    from_id, to_id, snap_candidates, road_graph
                )
                
                if is_straight or osm_path is None:
                    segment_lats = [nodes[from_id]['lat'], nodes[to_id]['lat']]
                    segment_lons = [nodes[from_id]['lon'], nodes[to_id]['lon']]
                else:
                    segment_lats = [osm_nodes[nid][0] for nid in osm_path]
                    segment_lons = [osm_nodes[nid][1] for nid in osm_path]
                
                if all_lats:
                    all_lats.extend(segment_lats[1:])
                    all_lons.extend(segment_lons[1:])
                else:
                    all_lats.extend(segment_lats)
                    all_lons.extend(segment_lons)
            
            # Draw alternate path with lighter blue
            fig.add_trace(go.Scattermap(
                lat=all_lats,
                lon=all_lons,
                mode='lines',
                line=dict(
                    width=5,
                    color='#ADD8E6'  # Light blue like Google Maps
                ),
                name=f'Alternate {idx + 1} (time: {alt_path["time"]:.1f})',
                hoverinfo='text',
                text=f'Alternate path {idx + 1}<br>Time: {alt_path["time"]:.1f}<br>Points: {len(all_lats)}',
                showlegend=True
            ))
    
    # Draw main path (fastest)
    if main_path:
        route_ids = main_path['nodes']
        print(f"\nMain path: {route_ids} (time: {main_path['time']})")
        
        if len(route_ids) >= 2:
            all_lats = []
            all_lons = []
            
            for i in range(len(route_ids) - 1):
                from_id = route_ids[i]
                to_id = route_ids[i + 1]
                
                osm_path, path_len, is_straight = find_best_path_between_nodes(
                    from_id, to_id, snap_candidates, road_graph
                )
                
                if is_straight or osm_path is None:
                    segment_lats = [nodes[from_id]['lat'], nodes[to_id]['lat']]
                    segment_lons = [nodes[from_id]['lon'], nodes[to_id]['lon']]
                else:
                    segment_lats = [osm_nodes[nid][0] for nid in osm_path]
                    segment_lons = [osm_nodes[nid][1] for nid in osm_path]
                
                if all_lats:
                    all_lats.extend(segment_lats[1:])
                    all_lons.extend(segment_lons[1:])
                else:
                    all_lats.extend(segment_lats)
                    all_lons.extend(segment_lons)
            
            # Draw main path with darker blue
            fig.add_trace(go.Scattermap(
                lat=all_lats,
                lon=all_lons,
                mode='lines',
                line=dict(
                    width=6,
                    color='#1E90FF'  # Dodger blue (main route)
                ),
                name=f'Main Route (time: {main_path["time"]:.1f})',
                hoverinfo='text',
                text=f'Main route<br>Time: {main_path["time"]:.1f}<br>Points: {len(all_lats)}',
                showlegend=True
            ))
    
    print(f"Drew {len(paths_list)} paths (1 main + {len(alternate_paths)} alternates)")