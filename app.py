import sys

import gradio as gr
import plotly.graph_objects as go

import pathing
from file_reader import parse_config_file

def main():
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

    def plot_map():
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
                showlegend=False
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
                zoom=18
            ),
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        return fig

    with gr.Blocks() as demo:
        gr.Markdown("# Map Visualization")
        map = gr.Plot()
        demo.load(plot_map, inputs=None, outputs=map)
    demo.launch()

if __name__ == "__main__":
    main()