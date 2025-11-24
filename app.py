import sys

import gradio as gr
import plotly.graph_objects as go

import pathing
from file_reader import parse_config_file

def main():
    nodes=[]
    ways=[]
    cameras=[]
    meta={}
    
    if(len(sys.argv)>1):
        nodes, ways, cameras, meta = parse_config_file(sys.argv[1])
    # print("Nodes:", nodes)
    # print("Ways:", ways)
    # print("Cameras:", cameras)
    # print("Meta:", meta)

    def plot_map():
        # Plotting the nodes/location on the map
        fig =go.Figure(go.Scattermap(
            lat=[nodes[n]['lat'] for n in nodes],
            lon=[nodes[n]['lon'] for n in nodes],
            mode='markers',
            marker=go.scattermap.Marker(
                size=9
            ),
            text=[f"ID: {n}, Label: {nodes[n]['label']}" for n in nodes],
        ))

        # #temporary generate way lines between nodes without considering road
        # from_node = '2'
        # to_node = '1'
        # fig.add_trace(go.Scattermap(
        #     lat=[nodes[from_node]['lat'], nodes[to_node]['lat']],
        #     lon=[nodes[from_node]['lon'], nodes[to_node]['lon']],
        #     mode='lines',
        #     line=go.scattermap.Line(
        #         width=2,
        #         color='blue'
        #     ),
        #     hoverinfo='none'
        # ))

        # For each way draw it onto the map
        for way in ways:
            pathing.draw_way(fig, nodes, way)

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
                zoom=9
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