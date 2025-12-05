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
path= {"nodes": [], "time": 0 }
paths= []

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
        info["lat"], info["lon"], osm_nodes, road_graph, k=1
    )
    print(f"Node {nid} ({info['label']}): snap candidates = {snap_candidates[nid][:3]}")


def plot_map(input_nodes, paths_list=None):
    """Plot the map with optional pathfinding visualization
    
    Args:
        input_nodes: Dictionary of nodes
        paths_list: List of path dicts [{"nodes": [...], "time": ...}] sorted by time
    """
    try:
        print(f"\n{'='*60}")
        print(f"plot_map called with {len(paths_list) if paths_list else 0} paths")
        print(f"{'='*60}\n")
        
        fig = go.Figure()

        # Draw OSM road network in the background
        pathing.draw_osm_roads(fig, osm_ways, osm_nodes, max_roads=50)

        # Draw dotted lines from assignment nodes to their snap points
        pathing.draw_snap_connections(fig, input_nodes, snap_candidates, osm_nodes)

        # Draw assignment ways if no paths provided
        show_assignment_ways = not paths_list or len(paths_list) == 0
        pathing.draw_assignment_ways(fig, ways, nodes, osm_nodes, snap_candidates, road_graph, show=show_assignment_ways)

        # Draw paths if provided
        pathing.draw_paths(fig, paths_list, nodes, osm_nodes, snap_candidates, road_graph)

        # Plot node markers
        for i, node in nodes.items():
            node_color = "black"
            if i == start:
                node_color = "green"
            elif i in goals:
                node_color = "red"
            fig.add_trace(go.Scattermap(
                lat=[node['lat']],
                lon=[node['lon']],
                mode='markers',
                marker=dict(size=12, color=node_color),
                text=f"{i}: {node['label']}<br>({node['lat']}, {node['lon']})",
                hoverinfo='text',
            ))

        fig.update_layout(
            autosize=True,
            hovermode='closest',
            height=800,
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
        
        print(f"\nMap plotting completed successfully\n")
        return fig
        
    except Exception as e:
        print(f"\nCRITICAL ERROR in plot_map: {e}")
        import traceback
        traceback.print_exc()
        
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
    
    # Controls - Store route as a State variable (now stores list of path dicts)
    route_state = gr.State(value=[])  # Empty list initially
    
    with gr.Row():
        plot_route_btn = gr.Button("Plot Route")
        clear_route_btn = gr.Button("Clear Route")
    
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
    def update_map(paths_list):
        """Update map with paths list
        
        Args:
            paths_list: List of path dicts [{"nodes": [1,2,6,13], "time": 15.5}, ...]
        """
        return plot_map(nodes, paths_list)
    
    plot_route_btn.click(
        fn=update_map,
        inputs=[route_state],  # Now expects list of path dicts
        outputs=[map_plot]
    )
    
    def clear_route():
        return [], plot_map(nodes, [])
    
    clear_route_btn.click(
        fn=clear_route,
        outputs=[route_state, map_plot]
    )

    demo.load(
        fn=lambda: plot_map(nodes, []),
        outputs=[map_plot]
    )

    # Example: Add a button to load example paths
    def load_example_paths():
        example_paths = [
            {"nodes": ["1", "2", "8", "9", "10", "9", "8", "2", "6", "13"], "time": 40.0},
            {"nodes": ["1", "14", "15", "12", "11", "9", "10", "9", "8", "2", "6", "13"], "time": 48.0},
            {"nodes": ["1", "2", "4", "5", "4", "2", "8", "9", "10", "9", "8", "2", "6", "13"], "time": 52.0},
        ]
        return example_paths, plot_map(nodes, example_paths)

    example_btn = gr.Button("Load Example Paths")
    example_btn.click(
        fn=load_example_paths,
        outputs=[route_state, map_plot]
    )

if __name__ == "__main__":
    demo.launch()