import sys
import xml.etree.ElementTree as ET

import gradio as gr
import plotly.graph_objects as go

import pathing
from file_reader import parse_config_file

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

def calculate_zoom_level(nodes):
    """Calculate optimal zoom level to fit all nodes in view"""
    if not nodes:
        return 12
    
    lats = [node['lat'] for node in nodes.values()]
    lons = [node['lon'] for node in nodes.values()]
    
    lat_range = max(lats) - min(lats)
    lon_range = max(lons) - min(lons)
    
    # Maximum range determines zoom
    max_range = max(lat_range, lon_range)
    
    # Approximate zoom levels based on lat/lon range
    # These values are approximate and may need adjustment
    if max_range >= 180:
        zoom = 1
    elif max_range >= 90:
        zoom = 2
    elif max_range >= 45:
        zoom = 3
    elif max_range >= 22.5:
        zoom = 4
    elif max_range >= 11.25:
        zoom = 5
    elif max_range >= 5.625:
        zoom = 6
    elif max_range >= 2.813:
        zoom = 7
    elif max_range >= 1.406:
        zoom = 8
    elif max_range >= 0.703:
        zoom = 9
    elif max_range >= 0.352:
        zoom = 10
    elif max_range >= 0.176:
        zoom = 11
    elif max_range >= 0.088:
        zoom = 12
    elif max_range >= 0.044:
        zoom = 13
    elif max_range >= 0.022:
        zoom = 14
    elif max_range >= 0.011:
        zoom = 15
    elif max_range >= 0.005:
        zoom = 16
    elif max_range >= 0.0025:
        zoom = 17
    elif max_range >= 0.00125:
        zoom = 18
    else:
        zoom = 19
    
    # Add some padding by reducing zoom slightly
    return max(1, zoom - 1)

with gr.Blocks() as demo:
    nodes=[]
    ways=[]
    cameras=[]
    start = None
    goals = []
    accident_multiplier = None
    # if(len(sys.argv)>1):
    nodes, ways, cameras, start, goals, accident_multiplier = parse_config_file("AI_AS2B/input.txt")
    
    # Load OSM road network (filtered to assignment area)
    osm_nodes, osm_ways = load_osm_graph("AI_AS2B/map.osm", nodes)
    print(f"Loaded {len(osm_nodes)} OSM nodes and {len(osm_ways)} OSM road segments (filtered)")
    
    # print("Nodes:", nodes)
    # print("Ways:", ways)
    # print("Cameras:", cameras)
    # print("Meta:", meta)
    
    def plot_map(input_nodes):
        fig =go.Figure()

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
            if idx > 2000:  # Limit to first 2000 roads for performance
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

        # Then draw assignment ways on top with normal colors
        prev_from = None
        prev_to = None
        for way in ways:
            if way['from'] == prev_to and way['to'] == prev_from:
                continue  #skip duplicate way
            pathing.draw_way(fig, nodes, way)
            prev_from = way['from']
            prev_to = way['to']

        #for each node plot a marker
        for i,node in nodes.items():
            # Figuring out the color of the node, Green for start nodes, Red for goal nodes, Black for normal nodes
            node_color="black"
            if i == start:
                node_color="green"
            elif i in goals:
                node_color="red"
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
                #showlegend=False
            ))

        # Set up the layout for the map
        fig.update_layout(
            autosize=True,
            hovermode='closest',
            map=dict(
                style="open-street-map",
                bearing=0,
                center=dict(
                    lat=sum(nodes[n]['lat'] for n in nodes)/len(nodes),
                    lon=sum(nodes[n]['lon'] for n in nodes)/len(nodes)
                ),
                pitch=0,
                zoom=calculate_zoom_level(nodes)
            ),
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        return fig
    
    #Drawing the actual app
    
    gr.Markdown("# Map Visualization")
    map = gr.Plot()

    #Nodes input table
    node_input=gr.DataFrame(
        value=[list(nodes[n].values()) for n in nodes],
        label="Nodes Data",
        headers=["lat", "lon", "label"],
        datatype=["number", "number", "text"],
        show_row_numbers=True,
        interactive=True,
        show_search="none")
    demo.load(plot_map, inputs=node_input, outputs=map)

if __name__ == "__main__":
    demo.launch()