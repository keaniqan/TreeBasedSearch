import sys

import gradio as gr
import plotly.graph_objects as go

import pathing
from file_reader import parse_config_file

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
    # print("Nodes:", nodes)
    # print("Ways:", ways)
    # print("Cameras:", cameras)
    # print("Meta:", meta)
    
    def plot_map(input_nodes):
        fig =go.Figure()

        # For each way draw it onto the map
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
                marker=go.scattermap.Marker(
                    size=10,
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
                bearing=0,
                center=dict(
                    lat= sum(nodes[n]['lat'] for n in nodes)/len(nodes),
                    lon= sum(nodes[n]['lon'] for n in nodes)/len(nodes)
                ),
                pitch=0,
                zoom=calculate_zoom_level(nodes)
            ),
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
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