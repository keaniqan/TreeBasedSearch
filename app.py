import sys
import xml.etree.ElementTree as ET

import gradio as gr
import plotly.graph_objects as go

import pathing
from file_reader import parse_config_file

# Global variables
nodes = []
ways = []
cameras = []
start = None
goals = []
accident_multiplier = None
osm_nodes = {}
osm_ways = []
road_graph = {}
snap_candidates = {}

def load_osm_graph(osm_path, assignment_nodes):
    """Load OSM file and extract road network within bounds of assignment nodes"""
    tree = ET.parse(osm_path)
    root = tree.getroot()

    # Get bounds from assignment nodes
    lats = [n['lat'] for n in assignment_nodes.values()]
    lons = [n['lon'] for n in assignment_nodes.values()]
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

# Load data
nodes, ways, cameras, start, goals, accident_multiplier = parse_config_file("AI_AS2B/input.txt")

# Load OSM road network (filtered to assignment area)
osm_nodes, osm_ways = load_osm_graph("AI_AS2B/map.osm", nodes)
print(f"Loaded {len(osm_nodes)} OSM nodes and {len(osm_ways)} OSM road segments (filtered)")

# Build road graph for pathfinding
road_graph = pathing.build_road_graph(osm_nodes, osm_ways)
print(f"Built road graph with {len(road_graph)} nodes")

# Precompute snap candidates for all assignment nodes
for nid, info in nodes.items():
    snap_candidates[nid] = pathing.k_nearest_graph_nodes(
        info["lat"], info["lon"], osm_nodes, road_graph, k=5
    )
    print(f"Node {nid} ({info['label']}): snap candidates = {snap_candidates[nid][:3]}")


def plot_map(input_nodes, route_nodes_str=""):
    """Plot the map with optional pathfinding visualization"""
    try:
        print(f"\n{'='*60}")
        print(f"plot_map called: route='{route_nodes_str}'")
        print(f"Sample way keys: {list(ways[0].keys()) if ways else 'NO WAYS'}")
        print(f"{'='*60}\n")
        
        fig = go.Figure()

        # First, draw OSM road network in the background with lighter colors
        osm_color_map = {
            "primary": "#87CEEB",
            "secondary": "#FFE4B5", 
            "tertiary": "#FFB6C1",
            "service": "#D3D3D3",
            "residential": "#E0E0E0",
            "unclassified": "#DCDCDC"
        }
        
        # Draw OSM roads as background (limit to avoid performance issues)
        for idx, osm_way in enumerate(osm_ways):
            if idx > 50:  # Increased limit for better coverage
                break
            lat1, lon1 = osm_nodes[osm_way['from_osm']]
            lat2, lon2 = osm_nodes[osm_way['to_osm']]
            
            # Determine color based on highway type
            highway_type = osm_way['highway_type'].lower()
            color = osm_color_map.get(highway_type, "#DCDCDC")
            
            # Draw OSM road segments with thinner lines
            fig.add_trace(go.Scattermap(
                lat=[lat1, lat2],
                lon=[lon1, lon2],
                mode='lines',
                line=dict(
                    width=1,
                    color=color
                ),
                hovertext=f"{osm_way['name']}<br>Type: {osm_way['highway_type']}",
                hoverinfo='text',
                showlegend=False,
                opacity=0.4
            ))

        # Then draw assignment ways
        prev_from = None
        prev_to = None
        
        print(f"\nDrawing {len(ways)} assignment ways...")
        
        for way_idx, way in enumerate(ways):
            try:
                if way['from'] == prev_to and way['to'] == prev_from:
                    continue  # skip duplicate way
                
                from_id = way['from']
                to_id = way['to']
                
                # Check if we should draw this as a road path or straight line
                # Draw roads following network when enabled
                print(f"\nWay {way_idx}: {from_id} → {to_id}")
                
                osm_path, path_len, is_straight = pathing.find_best_path_between_nodes(from_id, to_id, 
                                                                                      snap_candidates, road_graph)
                
                # Determine way color based on type
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
                    # Use straight line for this segment
                    segment_lats = [nodes[from_id]['lat'], nodes[to_id]['lat']]
                    segment_lons = [nodes[from_id]['lon'], nodes[to_id]['lon']]
                    
                    hovertext = (f"{way['id']}: {way['name']}<br>"
                                f"From: {from_id}<br>"
                                f"To: {to_id}<br>"
                                f"(Straight line - nodes too close)")
                    
                    print(f"  → Straight line with {len(segment_lats)} points")
                    
                    fig.add_trace(go.Scattermap(
                        lat=segment_lats,
                        lon=segment_lons,
                        mode='lines',
                        line=dict(
                            width=3,
                            color=way_color,
                            #dash='dash'
                        ),
                        hovertext=hovertext,
                        hoverinfo='text',
                        showlegend=False
                    ))
                else:
                    # Use road path - extract coordinates from OSM nodes
                    segment_lats = [osm_nodes[nid][0] for nid in osm_path]
                    segment_lons = [osm_nodes[nid][1] for nid in osm_path]
                    
                    hovertext = (f"{way['id']}: {way['name']}<br>"
                                f"From: {from_id}<br>"
                                f"To: {to_id}<br>"
                                f"Road distance: {path_len:.3f} km<br>"
                                f"OSM nodes: {len(osm_path)}")
                    
                    print(f"  → Road path with {len(segment_lats)} points")
                    print(f"     Start: ({segment_lats[0]:.6f}, {segment_lons[0]:.6f})")
                    print(f"     End: ({segment_lats[-1]:.6f}, {segment_lons[-1]:.6f})")
                    
                    # Draw the complete path for this way
                    fig.add_trace(go.Scattermap(
                        lat=segment_lats,
                        lon=segment_lons,
                        mode='lines',
                        line=dict(
                            width=4,
                            color=way_color
                        ),
                        hovertext=hovertext,
                        hoverinfo='text',
                        showlegend=False
                    ))
                
                prev_from = way['from']
                prev_to = way['to']
                
            except Exception as e:
                print(f"\n!!! ERROR drawing way {way_idx}: {e}")
                print(f"    Way data: {way}")
                import traceback
                traceback.print_exc()
                # Draw as straight line on error
                try:
                    pathing.draw_way(fig, nodes, way)
                except:
                    pass

        # Draw custom route if provided
        if route_nodes_str.strip():
            route_ids = [x.strip() for x in route_nodes_str.split(',') if x.strip()]
            print(f"\nPlotting custom route: {route_ids}")
            
            if len(route_ids) >= 2:
                # Validate all node IDs
                valid = all(nid in nodes for nid in route_ids)
                if valid:
                    # Collect all path segments
                    all_lats = []
                    all_lons = []
                    
                    for i in range(len(route_ids) - 1):
                        from_id = route_ids[i]
                        to_id = route_ids[i + 1]
                        
                        osm_path, path_len, is_straight = pathing.find_best_path_between_nodes(from_id, to_id, 
                                                                                              snap_candidates, road_graph)
                        
                        if is_straight or osm_path is None:
                            # Use straight line
                            segment_lats = [nodes[from_id]['lat'], nodes[to_id]['lat']]
                            segment_lons = [nodes[from_id]['lon'], nodes[to_id]['lon']]
                        else:
                            # Use road path
                            segment_lats = [osm_nodes[nid][0] for nid in osm_path]
                            segment_lons = [osm_nodes[nid][1] for nid in osm_path]
                        
                        # Append to route (skip first point if not first segment to avoid duplicates)
                        if all_lats:
                            all_lats.extend(segment_lats[1:])
                            all_lons.extend(segment_lons[1:])
                        else:
                            all_lats.extend(segment_lats)
                            all_lons.extend(segment_lons)
                    
                    # Draw the complete route
                    fig.add_trace(go.Scattermap(
                        lat=all_lats,
                        lon=all_lons,
                        mode='lines',
                        line=dict(
                            width=6,
                            color='red'
                        ),
                        name=f'Route: {" → ".join(route_ids)}',
                        hoverinfo='text',
                        text=f'Custom route with {len(all_lats)} points',
                        showlegend=True
                    ))
                    
                    print(f"Drew route with {len(all_lats)} total points")

        # For each node plot a marker
        for i, node in nodes.items():
            # Figure out the color of the node
            node_color = "black"
            if i == start:
                node_color = "green"
            elif i in goals:
                node_color = "red"
            fig.add_trace(go.Scattermap(
                lat=[node['lat']],
                lon=[node['lon']],
                mode='markers',
                marker=dict(
                    size=12,
                    color=node_color
                ),
                text=f"{i}: {node['label']}<br>({node['lat']}, {node['lon']})",
                hoverinfo='text',
            ))

        # Set up the layout for the map
        fig.update_layout(
            autosize=True,
            hovermode='closest',
            height=800,  # Add this line
            map=dict(
                style="open-street-map",
                bearing=0,
                center=dict(
                    lat=sum(nodes[n]['lat'] for n in nodes) / len(nodes),
                    lon=sum(nodes[n]['lon'] for n in nodes) / len(nodes)
                ),
                pitch=0,
                zoom=15
            ),
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
        
        print(f"\n{'='*60}")
        print(f"Map plotting completed successfully")
        print(f"{'='*60}\n")
        
        return fig
        
    except Exception as e:
        print(f"\n{'!'*60}")
        print(f"CRITICAL ERROR in plot_map: {e}")
        print(f"{'!'*60}")
        import traceback
        traceback.print_exc()
        
        # Return error figure
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error: {str(e)}<br>Check console for details",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return error_fig

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Map Visualization with Road Pathfinding")
    
    # Controls
    
    with gr.Row():
        route_input = gr.Textbox(
            label="Custom Route (comma-separated node IDs)",
            placeholder="e.g., 1,5,7,10",
            value=""
        )
        plot_route_btn = gr.Button("Plot Route")
        clear_route_btn = gr.Button("Clear Route")
    
    gr.Markdown("*Enter assignment node IDs in travel order from origin to destination*")
    
    # Map display
    with gr.Column():
        map_plot = gr.Plot(container=True)
    
    # Node input table
    node_input = gr.DataFrame(
        value=[list(nodes[n].values()) for n in nodes],
        label="Nodes Data",
        headers=["lat", "lon", "label"],
        datatype=["number", "number", "text"],
        show_row_numbers=True,
        interactive=True,
        show_search=False
    )
    
    # Event handlers
    def update_map(show_path, route_str):
        return plot_map(node_input, show_path, route_str)
    
    plot_route_btn.click(
        fn=update_map,
        inputs=[route_input],
        outputs=[map_plot]
    )
    
    def clear_route():
        return "", plot_map(node_input, False, "")
    
    clear_route_btn.click(
        fn=clear_route,
        outputs=[route_input, map_plot]
    )
    
    demo.load(
        fn=lambda: plot_map(node_input, ""),
        outputs=[map_plot]
    )

if __name__ == "__main__":
    demo.launch()