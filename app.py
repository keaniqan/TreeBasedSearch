import pandas as pd
import gradio as gr
import plotly.graph_objects as go
import file_reader
import pathing
import image_classification

from strategies.dijkstra import run_dijkstra

# ============================
# Initialization stuff
# ============================
# Load initial configuration
init_nodes_df, ways_df, cameras_df, start, goals, accident_multiplier = file_reader.parse_config_file("AI_AS2B\\input.txt")
image_classification.load_model("models\\best_finetuned.keras")

def pathFindingMap(nodes_df, ways_df, start, goals, accident_multiplier):
    fig = go.Figure()
    paths_df = pd.DataFrame(columns=['path', 'time'])

    #processing inputs
    start = int(start)
    goals = [int(g.strip()) for g in goals.split(",") if g.strip().isdigit()]
    accident_multiplier = float(accident_multiplier)

    # ===========================================
    # CAMERA PROCESSING
    # ===========================================
    # Update ways with accident severity from cameras
    for cam_id, camera in cameras_df.iterrows():
        way_id = camera['way_id']
        #skip if way id not in ways dataframe
        if way_id not in ways_df['id'].values:
            continue

        #use iamge_recognition module to get accident severity
        accident_severity = image_classification.clasify_accident(camera['image_path'])
        print(f"Camera {cam_id} on way {way_id} detected accident severity {accident_severity}")
        ways_df.loc[ways_df['id'] == way_id, 'accident_severity'] = accident_severity
        ways_df.loc[ways_df['id'] == way_id, 'final_time'] = ways_df.loc[ways_df['id'] == way_id, 'base_time'] * (1 + accident_severity * accident_multiplier)

    # ===========================================
    # PATH FINDING MAP PROCESSING
    # ===========================================
    temp_paths = []
    #using dijakstra to find path
    goal, nodes_created, path = run_dijkstra(nodes_df, ways_df, start, goals)
    if path is not None:
        temp_paths.append({
            'Goal': goal,
            'nodes': path,
            'time': sum(
                ways_df[
                    ( (ways_df['from'] == path[i]) & (ways_df['to'] == path[i+1]) ) |
                    ( (ways_df['from'] == path[i+1]) & (ways_df['to'] == path[i]) )
                ]['final_time'].values[0]
                for i in range(len(path)-1)
            )
        })
    # Populate paths dataframe for display
    paths_df = pd.DataFrame(temp_paths, columns=['Goal', 'nodes', 'time'])
    paths_df = paths_df.sort_values(by='time').reset_index(drop=True)
    paths_df['nodes'] = paths_df['nodes'].apply(lambda p: " -> ".join(str(n) for n in p))
    paths_df.rename(columns={'nodes': 'Path', 'time': 'Total Time (mins)'}, inplace=True)
    

    #OSM road network initilization
    osm_nodes, osm_ways =  pathing.load_osm_file("AI_AS2B\\map.osm", nodes_df)
    osm_graph = pathing.build_road_graph(osm_nodes, osm_ways)
    snap_candidates = {}
    for nid, node in nodes_df.iterrows():
        snap_candidates[nid] = pathing.k_nearest_graph_nodes(
            node["lat"], node["lon"], osm_nodes, osm_graph, k=1
        )

    # Draw ways
    pathing.draw_snap_connections(fig, nodes_df, snap_candidates, osm_nodes)
    pathing.draw_assignment_ways(fig, ways_df, nodes_df, osm_nodes, snap_candidates, osm_graph)

    #Drawing the actual paths found
    pathing.draw_paths(fig, temp_paths, nodes_df, osm_nodes, snap_candidates, osm_graph)

    # Draw nodes onto the map
    for idx, node in nodes_df.iterrows():
        if idx == start:
            node_color = "green"
        elif idx in goals:
            node_color = "red"
        else:
            node_color = "blue"
        
        fig.add_trace(go.Scattermap(
            lat=[node['lat']],
            lon=[node['lon']],
            mode='markers',
            marker=dict(
                size=12,
                color=node_color
            ),
            text="[" + str(idx) + "] " + node['label'],
            hoverinfo='text',
            name="[" + str(idx) + "] " + node['label'],
            showlegend=False
        ))
    
    # Set up the layout for the map
    fig.update_layout(
        autosize=True,
        hovermode='closest',
        map=dict(
            style="open-street-map",
            bearing=0,
            center=dict(
                lat=sum(nodes_df['lat']) / len(nodes_df),
                lon=sum(nodes_df['lon']) / len(nodes_df)
            ),
            pitch=0,
            zoom=15
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    return fig,ways_df, paths_df
        
#gradio interface
with gr.Blocks() as demo:
    with gr.Column():
        map = gr.Plot()
        paths_out = gr.DataFrame(interactive=False, label="Found Paths")
        with gr.Row():
            inp_start = gr.Number(value=start, label="Start Node ID", interactive=True)
            inp_goals = gr.Textbox(value=", ".join(goals), label="Goal Node IDs (comma separated)", interactive=True)
        inp_accident_multiplier = gr.Number(value=accident_multiplier, label="Accident Multiplier", interactive=True)
        btn = gr.Button(value="Generate path")
    with gr.Tab("Nodes"):
        nodes = gr.Dataframe(
            value=init_nodes_df, label="Nodes", interactive=True, datatype=["number", "number", "text"]
        )
    with gr.Tab("Ways"):
        ways = gr.Dataframe(
            value=ways_df, label="Ways", interactive=True, datatype=["number", "number", "number", "text", "text", "number", "text", "number"]
        )
    demo.load(pathFindingMap,[nodes, ways, inp_start, inp_goals, inp_accident_multiplier],[map,ways, paths_out])
    btn.click(pathFindingMap,[nodes, ways, inp_start, inp_goals, inp_accident_multiplier],[map,ways, paths_out])
if __name__ == "__main__":
    demo.launch()