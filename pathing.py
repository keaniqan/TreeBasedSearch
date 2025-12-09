import heapq
import plotly.graph_objects as go
import xml.etree.ElementTree as ET
import constants

def draw_line_ways(fig, ways_df, nodes_df):
    """Draw simple straight line ways on the map for visualization
    
    Args:
        fig: Plotly figure object
        ways_df: DataFrame of ways
        nodes_df: DataFrame of nodes
    """
    for way_idx, way in ways_df.iterrows():
        from_id = way['from']
        to_id = way['to']
        
        segment_lats = [nodes_df.loc[from_id]['lat'], nodes_df.loc[to_id]['lat']]
        segment_lons = [nodes_df.loc[from_id]['lon'], nodes_df.loc[to_id]['lon']]
        
        fig.add_trace(go.Scattermap(  # Changed from Scattermapbox
            lat=segment_lats,
            lon=segment_lons,
            mode='lines',
            line=dict(width=2, color='blue'),
            showlegend=False
        ))

def draw_line_path(fig, path_node_ids, nodes_df):
    """Draw a simple straight line path on the map for visualization
    
    Args:
        fig: Plotly figure object
        path_node_ids: List of node IDs representing the path
        nodes_df: DataFrame of nodes
    """
    path_lats = [nodes_df.loc[nid]['lat'] for nid in path_node_ids]
    path_lons = [nodes_df.loc[nid]['lon'] for nid in path_node_ids]
    
    fig.add_trace(go.Scattermap(  # Changed from Scattermapbox
        lat=path_lats,
        lon=path_lons,
        mode='lines',
        line=dict(width=4, color='red'),
        showlegend=False
    ))
def load_osm_file(osm_path, nodes_df):
    """Load OSM file and extract road network within bounds of assignment nodes"""
    tree = ET.parse(osm_path)
    root = tree.getroot()

    # Get bounds from assignment nodes  
    lats = nodes_df['lat'].tolist()
    lons = nodes_df['lon'].tolist()
    min_lat, max_lat = min(lats) - 0.005, max(lats) + 0.005
    min_lon, max_lon = min(lons) - 0.005, max(lons) + 0.005

    osm_nodes = {}
    for n in root.findall("node"):
        nid = n.attrib["id"]
        lat = float(n.attrib["lat"])
        lon = float(n.attrib["lon"])
        # Only include nodes within bounds
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            osm_nodes[nid] = (lat, lon)

    osm_ways = []
    for w in root.findall("way"):
        nd_refs = [nd.attrib["ref"] for nd in w.findall("nd")]
        tags = {t.attrib.get("k"): t.attrib.get("v") for t in w.findall("tag")}
        if "highway" not in tags:
            continue
        
        highway_type = tags.get("highway", "")
        name = tags.get("name", "Unnamed Road")
        
        # Store way information with node references
        for i in range(len(nd_refs) - 1):
            a, b = nd_refs[i], nd_refs[i+1]
            if a in osm_nodes and b in osm_nodes:
                osm_ways.append({
                    "from_osm": a,
                    "to_osm": b,
                    "highway_type": highway_type,
                    "name": name
                })

    return osm_nodes, osm_ways

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

def draw_snap_connections(fig, nodes_df, snap_candidates, osm_nodes):
    """Draw dotted lines from assignment nodes to their nearest snap points
    
    Args:
        fig: Plotly figure object
        nodes: Dictionary of assignment nodes
        snap_candidates: Dictionary mapping node IDs to their snap candidates
        osm_nodes: Dictionary of OSM nodes
    """
    print(f"\nDrawing snap connections...")
    
    for node_id, node in nodes_df.iterrows():
        if node_id in snap_candidates and snap_candidates[node_id]:
            nearest_snap = snap_candidates[node_id][0]  # Get nearest snap point
            snap_lat, snap_lon = osm_nodes[nearest_snap]
            
            # Create dotted effect by drawing multiple small segments
            num_dots = 10
            lats_dotted = []
            lons_dotted = []
            
            for j in range(num_dots):
                t = j / (num_dots - 1)
                lat_interp = node['lat'] + t * (snap_lat - node['lat'])
                lon_interp = node['lon'] + t * (snap_lon - node['lon'])
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

def draw_assignment_ways(fig, ways_df, nodes_df, osm_nodes, snap_candidates, road_graph, show=True):
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
    prev_from = None
    prev_to = None
    
    for way_id, way in ways_df.iterrows():
        try:
            if way['from'] == prev_to and way['to'] == prev_from:
                continue
            
            from_id = way['from']
            to_id = way['to']
            
            print(f"\nWay {way_id}: {from_id} → {to_id}")
            
            osm_path, path_len, is_straight = find_best_path_between_nodes(
                from_id, to_id, snap_candidates, road_graph
            )
            
            way_color = "black"
            if way['type'] == "primary":
                way_color = constants.PRIMARY_ROAD_COLOR
            elif way['type'] == "secondary":
                way_color = constants.SECONDARY_ROAD_COLOR
            elif way['type'] == "tertiary":
                way_color = constants.TERTIARY_ROAD_COLOR
            elif way['type'] == "service":
                way_color = constants.SERVICE_ROAD_COLOR
            
            if is_straight or osm_path is None:
                segment_lats = [nodes_df.at[from_id, 'lat'], nodes_df.at[to_id, 'lat']]
                segment_lons = [nodes_df.at[from_id, 'lon'], nodes_df.at[to_id, 'lon']]
            else:
                segment_lats = [osm_nodes[nid][0] for nid in osm_path]
                segment_lons = [osm_nodes[nid][1] for nid in osm_path]

                
            #Formating hovertext for way type, accident severity and times
            hovertext = (
                f"<b>{way['id']}: {way['name']}</b><br>"
                f"<b>Type</b>: {way['type']}<br>"
                f"<b>Accident Severity</b>: {way['accident_severity']}<br>"
                f"<b>Time</b>: {way['final_time']} mins<br>")
            
            fig.add_trace(go.Scattermap(
                lat=segment_lats,
                lon=segment_lons,
                mode='lines',
                line=dict(width=constants.ROAD_LINE_WIDTH if not is_straight else constants.ROAD_LINE_WIDTH-1, color=way_color),
                hovertext=hovertext,
                hoverinfo='text',
                showlegend=False
            ))
            
            prev_from = way['from']
            prev_to = way['to']
            
        except Exception as e:
            print(f"\n!!! ERROR drawing way {way_id}: {e}")
            import traceback
            traceback.print_exc()

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

def draw_paths(fig, paths_list, nodes_df,ways_df, osm_nodes, snap_candidates, road_graph):
    """Draw main and alternate paths on the map
    
    Args:
        fig: Plotly figure object
        paths_list: List of path dicts [{"path_name": ..., "goal": ..., "nodes": [...], "time": ...}]
        nodes_df: DataFrame of assignment nodes
        ways_df: DataFrame of assignment ways
        osm_nodes: Dictionary of OSM nodes
        snap_candidates: Dictionary mapping node IDs to snap candidates
        road_graph: Road graph for pathfinding
    """
    if not paths_list or len(paths_list) == 0:
        return
    
    #Sort path by time (ascending)
    paths_list = sorted(paths_list, key=lambda p: p['time'])

    def draw(path, path_index):
        if(path['time'] == float('inf')):
            return  # Skip paths with no valid route

        node_ids = path['nodes']
        print(f"\nPath {path_index}: {node_ids} (time: {path['time']})")

        #Generate lat/lon lists for the full path
        for i in range(len(node_ids) - 1):
            from_id = node_ids[i]
            to_id = node_ids[i + 1]

            #Retrieve the way from ways_df using from and to
            ways = ways_df[(ways_df['from'] == from_id) & (ways_df['to'] == to_id)].iloc[0]
            
            # Fiding the best osm path between the two nodes and generate lat/lon segments
            lats = []
            lons = []
            osm_path, path_len, is_straight = find_best_path_between_nodes(from_id, to_id, snap_candidates, road_graph)
            if is_straight or osm_path is None:
                segment_lats = [nodes_df.loc[from_id, 'lat'], nodes_df.loc[to_id, 'lat']]
                segment_lons = [nodes_df.loc[from_id, 'lon'], nodes_df.loc[to_id, 'lon']]
            else:
                segment_lats = [osm_nodes[nid][0] for nid in osm_path]
                segment_lons = [osm_nodes[nid][1] for nid in osm_path]
            if lats:
                lats.extend(segment_lats[1:])
                lons.extend(segment_lons[1:])
            else:
                lats.extend(segment_lats)
                lons.extend(segment_lons)
        
            #Generate the line color. If there was severe accident on the way, color it red, if it was moderate, color it orange, else use default road color.
            #An alternate palate is used for alternate paths to differentiate them from the main path.
            if path_index == 0:
                if ways['accident_severity'] == 3:
                    line_color = constants.PRIMARY_PATH_SEVERE_COLOR
                elif ways['accident_severity'] == 2:
                    line_color = constants.PRIMARY_PATH_MODERATE_COLOR
                elif ways['accident_severity'] == 1:
                    line_color = constants.PRIMARY_PATH_MINOR_COLOR
                else:
                    line_color = constants.PRIMARY_PATH_COLOR
            else:
                if ways['accident_severity'] == 3:
                    line_color = constants.SECONDARY_PATH_COLOR_SEVERE_COLOR
                elif ways['accident_severity'] == 2:
                    line_color = constants.SECONDARY_PATH_COLOR_MODERATE_COLOR
                elif ways['accident_severity'] == 1:
                    line_color = constants.SECONDARY_PATH_COLOR_MINOR_COLOR
                else:
                    line_color = constants.SECONDARY_PATH_COLOR
            fig.add_trace(go.Scattermap(
                lat=lats,
                lon=lons,
                mode='lines',
                line=dict(
                    width=constants.PATH_LINE_WIDTH if path_index == 0 else constants.PATH_LINE_WIDTH-1,
                    color=line_color
                ),
                name=f'{path["path_name"]}(Time: {path["time"]:.1f})',
                legendgroup=f'{path["path_name"]}(Time: {path["time"]:.1f})',
                hoverinfo='text',
                text=f'{path["path_name"]}<br>Time: {path["time"]:.1f}',
                showlegend=(i == 0)
            ))
    
    # Draw alternate paths first (so main path appears on top)
    for idx, alt_path in enumerate(paths_list[1:]):
        draw(alt_path, idx+1)
    draw(paths_list[0], 0)  # Draw main path last
    
    print(f"Drew {len(paths_list)} paths (1 main + {len(paths_list) - 1} alternates)")