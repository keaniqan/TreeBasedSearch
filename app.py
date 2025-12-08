import pandas as pd
import gradio as gr
import plotly.graph_objects as go
import file_reader
import pathing
import image_classification
import os

from strategies.dfs import run_dfs
from strategies.bfs import run_bfs
from strategies.gbfs import run_gbfs
from strategies.astar import run_astar
from strategies.dijkstra import run_dijkstra
from strategies.beam import run_beam

# ============================
# Initialization stuff
# ============================
# Get list of available test case files
TEST_CASES_FOLDER = "Test_Cases_Map"
available_files = [f for f in os.listdir(TEST_CASES_FOLDER) if f.endswith('.txt')]
default_file = "default.txt" if "default.txt" in available_files else available_files[0]

# Load initial configuration
init_nodes_df, ways_df, cameras_df, start, goals, accident_multiplier = file_reader.parse_config_file(os.path.join(TEST_CASES_FOLDER, default_file))
image_classification.load_model("models\\best_finetuned.keras") 

def load_config_file(filename):
    """Load a configuration file and return the data"""
    filepath = os.path.join(TEST_CASES_FOLDER, filename)
    nodes_df, ways_df, cameras_df, start, goals, accident_multiplier = file_reader.parse_config_file(filepath)
    return nodes_df, ways_df, start, ", ".join(map(str, goals)), accident_multiplier

def pathFindingMap(nodes_df, ways_df, start, goals, accident_multiplier, is_show_ways=False):
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
    def add_path(path_name:str, goal: int, path: list[int]):
        if path is None:
            temp_paths.append({
                'path_name': path_name,
                'goal': goal,
                'nodes': [],
                'time': float('inf')
            })
            return
        temp_paths.append({
            'path_name': path_name,
            'goal': goal,
            'nodes': path,
            'time': sum(
                ways_df[
                    ( (ways_df['from'] == path[i]) & (ways_df['to'] == path[i+1]) ) |
                    ( (ways_df['from'] == path[i+1]) & (ways_df['to'] == path[i]) )
                ]['final_time'].values[0]
                for i in range(len(path)-1)
            )
        })
    #Using A* to find path
    goal, nodes_created, path = run_astar(nodes_df, ways_df, start, goals)
    add_path("A*", goal, path)
    #Using Beam Search to find path
    goal, nodes_created, path = run_beam(nodes_df, ways_df, start, goals, beam_width=3)
    add_path("Beam Search", goal, path)
    #Using BFS to find path
    goal, nodes_created, path = run_bfs(nodes_df, ways_df, start, goals)
    add_path("BFS", goal, path)
    #Using DFS to find path
    goal, nodes_created, path = run_dfs(nodes_df, ways_df, start, goals)
    add_path("DFS", goal, path)
    #using dijakstra to find path
    goal, nodes_created, path = run_dijkstra(nodes_df, ways_df, start, goals)
    add_path("Dijkstra", goal, path)
     #Using GBFS to find path
    goal, nodes_created, path = run_gbfs(nodes_df, ways_df, start, goals)
    add_path("GBFS", goal, path)

    # Populate paths dataframe for display
    paths_df = pd.DataFrame(temp_paths, columns=['path_name','goal', 'nodes', 'time'])
    paths_df = paths_df.sort_values(by='time').reset_index(drop=True)
    paths_df['nodes'] = paths_df['nodes'].apply(lambda p: " -> ".join(str(n) for n in p))
    paths_df.replace(float('inf'), 'No Path Found', inplace=True)
    paths_df.rename(columns={'path_name':'Path Name', 'goal': 'Goal', 'nodes': 'Path', 'time': 'Total Time (mins)'}, inplace=True)
    

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
    if is_show_ways:
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
            node_color = "black"
        
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
        height=800,
        map=dict(
            style="open-street-map",
            bearing=0,
            center=dict(
                lat=sum(nodes_df['lat']) / len(nodes_df),
                lon=sum(nodes_df['lon']) / len(nodes_df)
            ),
            pitch=0,
            zoom=15.5,    
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(
            itemwidth=30,
            y=0.9)
    )
    return fig,ways_df, paths_df

def clearMap(nodes_df, ways_df, start, goals, accident_multiplier, is_show_ways=False):
    """Create a map with only nodes, no paths"""
    fig = go.Figure()
    
    #processing inputs
    start = int(start)
    goals = [int(g.strip()) for g in goals.split(",") if g.strip().isdigit()]
    
    #OSM road network initilization
    osm_nodes, osm_ways =  pathing.load_osm_file("AI_AS2B\\map.osm", nodes_df)
    osm_graph = pathing.build_road_graph(osm_nodes, osm_ways)
    snap_candidates = {}
    for nid, node in nodes_df.iterrows():
        snap_candidates[nid] = pathing.k_nearest_graph_nodes(
            node["lat"], node["lon"], osm_nodes, osm_graph, k=1
        )

    # Draw ways if checkbox is enabled
    pathing.draw_snap_connections(fig, nodes_df, snap_candidates, osm_nodes)
    if is_show_ways:
        pathing.draw_assignment_ways(fig, ways_df, nodes_df, osm_nodes, snap_candidates, osm_graph)

    # Draw nodes onto the map
    for idx, node in nodes_df.iterrows():
        if idx == start:
            node_color = "green"
        elif idx in goals:
            node_color = "red"
        else:
            node_color = "black"
        
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
        height=800,
        map=dict(
            style="open-street-map",
            bearing=0,
            center=dict(
                lat=sum(nodes_df['lat']) / len(nodes_df),
                lon=sum(nodes_df['lon']) / len(nodes_df)
            ),
            pitch=0,
            zoom=15.5,    
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(
            itemwidth=30,
            y=0.9)
    )
    
    # Return empty dataframe for paths
    empty_paths = pd.DataFrame(columns=['Path Name', 'Goal', 'Path', 'Total Time (mins)'])
    return fig, ways_df, empty_paths

def load_and_generate(filename, is_show_ways=False):
    """Load a configuration file and generate the path automatically"""
    filepath = os.path.join(TEST_CASES_FOLDER, filename)
    nodes_df, ways_df, cameras_df, start, goals, accident_multiplier = file_reader.parse_config_file(filepath)
    goals_str = ", ".join(map(str, goals))
    
    # Generate the map with paths
    fig, ways_df, paths_df = pathFindingMap(nodes_df, ways_df, start, goals_str, accident_multiplier, is_show_ways)
    
    return nodes_df, ways_df, start, goals_str, accident_multiplier, fig, paths_df

#gradio interface
with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            map = gr.Plot()
        with gr.Row():
            paths_out = gr.DataFrame(interactive=False, label="Found Paths")
        with gr.Row():
            file_dropdown = gr.Dropdown(choices=available_files, value=default_file, label="Select Test Case File", interactive=True)
        with gr.Row():
            inp_start = gr.Number(value=start, label="Start Node ID", interactive=True)
            inp_goals = gr.Textbox(value=", ".join(goals), label="Goal Node IDs (comma separated)", interactive=True)
            inp_accident_multiplier = gr.Number(value=accident_multiplier, label="Accident Multiplier", interactive=True)
        with gr.Row():
            is_show_ways = gr.Checkbox(value=True, label="Show Ways", interactive=True)
        with gr.Row():
            btn = gr.Button(value="Generate path")
            btn_clear = gr.Button(value="Clear Routes")
    with gr.Tab("Nodes"):
        nodes = gr.Dataframe(
            value=init_nodes_df, interactive=True, datatype=["number", "number", "text"]
        )
    with gr.Tab("Ways"):
        ways = gr.Dataframe(
            value=ways_df, interactive=True, datatype=["number", "number", "number", "text", "text", "number", "text", "number"]
        )
    with gr.Tab("Cameras"):
        for camers in cameras_df.itertuples():
            with gr.Row():
                gr.Textbox(value=str(camers.Index), label="Camera ID", interactive=False)
                gr.Number(value=camers.way_id, label="Way ID", interactive=True)
                gr.Image(value=camers.image_path, label="Camera Image", interactive=True)
    
    # When file dropdown changes, load and automatically generate path
    file_dropdown.change(
        load_and_generate,
        inputs=[file_dropdown, is_show_ways],
        outputs=[nodes, ways, inp_start, inp_goals, inp_accident_multiplier, map, paths_out]
    )
    
    demo.load(pathFindingMap,[nodes, ways, inp_start, inp_goals, inp_accident_multiplier, is_show_ways],[map,ways, paths_out])
    btn.click(pathFindingMap,[nodes, ways, inp_start, inp_goals, inp_accident_multiplier, is_show_ways],[map,ways, paths_out])
    btn_clear.click(clearMap,[nodes, ways, inp_start, inp_goals, inp_accident_multiplier, is_show_ways],[map,ways, paths_out])

if __name__ == "__main__":
    demo.launch()