import pandas as pd
import gradio as gr
import plotly.graph_objects as go
import file_reader
import pathing
import image_classification
import os
import constants
import argparse

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
available_files = [f for f in os.listdir(constants.TEST_CASE_FOLDER) if f.endswith('.txt')]

# Dataframe definitions, these will be initially be populated with the data of the config file parse
init_nodes_df = pd.DataFrame(columns = ['id','lat', 'lon', 'label'], index=['id'])
init_ways_df = pd.DataFrame(columns = ['id', 'from', 'to', 'name', 'type', 'base_time', 'accident_severity', 'final_time'], index=['id'])
init_cameras_df = pd.DataFrame(columns = ['way_id', 'image_path', 'accident_severity', 'predictions'])
init_start = 0
init_goals = []
init_accident_multiplier = 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch gradio path finding application")
    parser.add_argument('--config', help='Initial configuration file', default='Test_Cases_Map\\default.txt')
    
    args = parser.parse_args()
    config = args.config
    init_nodes_df, init_ways_df, init_cameras_df, init_start, init_goals, init_accident_multiplier = file_reader.parse_config_file(config)

    #Update default file to dislplay the currently loaded file
    default_file = os.path.basename(config)
image_classification.load_model(constants.ENUM_AI_MODELS[0])

def pathFindingMap(nodes_df, ways_df, cameras_df,start, goals, accident_multiplier, is_show_ways=False, is_show_paths=True):
    #Indexing dataframes properly
    nodes_df = nodes_df.astype({'id': 'int'})
    ways_df = ways_df.astype({'id': 'int', 'from': 'int', 'to': 'int', 'base_time': 'float', 'final_time': 'float', 'accident_severity': 'float'})
    nodes_df.set_index('id', inplace=True, drop=False)
    ways_df.set_index('id', inplace=True, drop=False)

    fig = go.Figure()
    paths_df = pd.DataFrame(columns=['path', 'time'])

    #processing inputs
    start = int(start)
    if not isinstance(goals, list):
        goals = [int(g.strip()) for g in goals.split(",") if g.strip().isdigit()]
    accident_multiplier = float(accident_multiplier)

    # ===========================================
    # CAMERA PROCESSING
    # ===========================================
    # Update ways with accident severity from cameras
    for cam_id, camera in cameras_df.iterrows():
        way_id = camera['way_id']
        #skip if way id not in ways dataframe
        if way_id not in ways_df.index:
            continue

        #use iamge_recognition module to get accident severity
        accident_severity, predictions = image_classification.clasify_accident(camera['image_path'])
        cameras_df.at[cam_id, 'accident_severity'] = accident_severity
        cameras_df.at[cam_id, 'predictions'] = predictions.tolist()
        ways_df.at[way_id, 'accident_severity'] = accident_severity
        ways_df.at[way_id, 'final_time'] = ways_df.at[way_id, 'base_time'] * (1 + accident_severity * accident_multiplier)

    # ===========================================
    # PATH FINDING MAP PROCESSING
    # ===========================================
    temp_paths = []
    def add_path(path_name:str, res):
        if res is None:
            temp_paths.append({
                'path_name': path_name,
                'goal': None,
                'nodes': [],
                'time': float('inf')
            })
            return
        goal, nodes_created, path = res
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
    add_path("A*", run_astar(nodes_df, ways_df, start, goals))
    #Using Beam Search to find path
    add_path("Beam Search", run_beam(nodes_df, ways_df, start, goals, beam_width=3))
    #Using BFS to find path
    add_path("BFS", run_bfs(nodes_df, ways_df, start, goals))
    #Using DFS to find path
    add_path("DFS", run_dfs(nodes_df, ways_df, start, goals))
    #using dijakstra to find path
    add_path("Dijkstra", run_dijkstra(nodes_df, ways_df, start, goals))
     #Using GBFS to find path
    add_path("GBFS", run_gbfs(nodes_df, ways_df, start, goals))

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
    for node_id, node in nodes_df.iterrows():
        snap_candidates[node_id] = pathing.k_nearest_graph_nodes(
            node["lat"], node["lon"], osm_nodes, osm_graph, k=1
        )

    # Draw ways
    pathing.draw_snap_connections(fig, nodes_df, snap_candidates, osm_nodes)
    if is_show_ways:
        pathing.draw_assignment_ways(fig, ways_df, nodes_df, osm_nodes, snap_candidates, osm_graph)

    #Drawing the actual paths found
    if is_show_paths:
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

    #hiding and showing the appropriate cameras
    camera_rows = [gr.Row(visible=False) for _ in range(constants.MAX_CAMERA_COUNT)]
    camera_severity_updates = [gr.Number(visible=False) for _ in range(constants.MAX_CAMERA_COUNT)]
    camera_predictions_updates = [gr.Textbox(visible=False) for _ in range(constants.MAX_CAMERA_COUNT)]
    camera_way_updates = [gr.Number(visible=False) for _ in range(constants.MAX_CAMERA_COUNT)]
    camera_image_updates = [gr.Image(visible=False) for _ in range(constants.MAX_CAMERA_COUNT)]
    row_id=0
    for cam_id, camera in cameras_df.iterrows():
        camera_rows[row_id] = gr.Row(visible=True)
        camera_way_updates[row_id] = gr.Number(visible=True, value=camera['way_id'])
        camera_severity_updates[row_id] = gr.Number(visible=True, value=camera['accident_severity'])
        camera_predictions_updates[row_id] = gr.Textbox(visible=True, value=", ".join([f"{x:.2f}" for x in camera['predictions']]))
        camera_image_updates[row_id] = gr.Image(visible=True, value=camera['image_path'])
        row_id+=1
    return [fig,nodes_df,ways_df, cameras_df, paths_df]+camera_rows+camera_way_updates+camera_severity_updates+camera_predictions_updates+camera_image_updates

def load_and_generate(filename, is_show_ways=False, is_show_paths=True):
    """Load a configuration file and generate the path automatically"""
    filepath = os.path.join(constants.TEST_CASE_FOLDER, filename)
    nodes_df, ways_df, cameras_df, start, goals, accident_multiplier = file_reader.parse_config_file(filepath)
    # Generate the map with paths
    return pathFindingMap(nodes_df, ways_df,cameras_df, start, goals, accident_multiplier, is_show_ways, is_show_paths)

def add_new_camera(cameras_df, image_path, way_id, model_name):
    """Add a new camera to the cameras dataframe"""
    if image_path is None:
        return cameras_df
    
    # Load the selected model
    image_classification.load_model(model_name)
    
    # Create new camera entry
    new_camera = pd.DataFrame([{
        'way_id': int(way_id),
        'image_path': image_path,
        'accident_severity': 0,
        'predictions': []
    }])
    
    # Append to cameras dataframe
    cameras_df = pd.concat([cameras_df, new_camera], ignore_index=True)
    
    return cameras_df

def delete_camera(cameras_df, camera_index):
    """Delete a camera from the cameras dataframe"""
    if camera_index < len(cameras_df):
        cameras_df = cameras_df.drop(cameras_df.index[camera_index]).reset_index(drop=True)
    return cameras_df


#================================================
#   GADIO INTERFACE
#================================================
#globals for dynamic gradio components
camera_rows = []
camera_way_rows = []
camera_severity_rows = []
camera_predictions_rows = []
camera_image_rows = []
camera_delete_btns = []
#App interface
with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            map = gr.Plot()
        with gr.Row():
            paths_out = gr.DataFrame(interactive=False, label="Found Paths")
        with gr.Row():
            is_show_ways = gr.Checkbox(value=True, label="Show Ways", interactive=True)
            is_show_paths = gr.Checkbox(value=True, label="Show Paths", interactive=True)
        with gr.Row():
            btn = gr.Button(value="Generate path")
        with gr.Row():
            file_dropdown = gr.Dropdown(choices=available_files, value=default_file, label="Select Test Case File", interactive=True)
            inp_ai_model = gr.Dropdown(choices=constants.ENUM_AI_MODELS, value=constants.ENUM_AI_MODELS[0], label="AI Model for Image Classification", interactive=True)
        with gr.Row():
            inp_start = gr.Number(value=init_start, label="Start Node ID", interactive=True)
            inp_goals = gr.Textbox(value=", ".join(str(num) for num in init_goals),label="Goal Node IDs (comma separated)", interactive=True)
            inp_accident_multiplier = gr.Number(value=init_accident_multiplier, label="Accident Multiplier", interactive=True)
    with gr.Tab("Nodes"):
        nodes = gr.Dataframe(
            value=init_nodes_df, interactive=True, datatype=["number", "number","number","text"]
        )
    with gr.Tab("Ways"):
        ways = gr.Dataframe(
            value=init_ways_df, interactive=True, datatype=["number", "number", "number", "text", "text", "number", "text", "number"]
        )
    with gr.Tab("Cameras"):
        with gr.Accordion("Add New Camera", open=True):
            with gr.Row():
                new_camera_image = gr.Image(type="filepath", label="Upload Car Image", interactive=True)
                with gr.Column():
                    new_camera_way = gr.Number(value=0, label="Select Way ID", interactive=True)
                    new_camera_model = gr.Dropdown(choices=constants.ENUM_AI_MODELS, value=constants.ENUM_AI_MODELS[0], label="Select AI Model", interactive=True)
                    add_camera_btn = gr.Button("Add Camera")
        
        for i in range(constants.MAX_CAMERA_COUNT):
            with gr.Row() as cam_row:
                with gr.Row():
                    way = gr.Number(value=0, label="Way ID", interactive=False)
                    severity = gr.Number(value=0, label="Accident Severity", interactive=False)
                    predictions = gr.Textbox(value="[]", label="Predictions (none, minor, moderate, severe)",min_width=30, interactive=False)
                image = gr.Image(type="filepath", label="Image Path", interactive=False)
                delete_btn = gr.Button("Delete", size="sm", variant="stop")
                camera_rows.append(cam_row)
            camera_way_rows.append(way)
            camera_severity_rows.append(severity)
            camera_predictions_rows.append(predictions)
            camera_image_rows.append(image)
            camera_delete_btns.append(delete_btn)
        inp_camera = gr.Dataframe(value=init_cameras_df, interactive=False,visible=True, max_height=1)

    # Event listeners
    file_dropdown.change(
        load_and_generate,
        inputs=[file_dropdown, is_show_ways, is_show_paths],
        outputs=[map,nodes,ways,inp_camera, paths_out]+camera_rows+camera_way_rows+camera_severity_rows+camera_predictions_rows+camera_image_rows
    )
    inp_ai_model.change(image_classification.load_model, inp_ai_model, None)
    add_camera_btn.click(
        add_new_camera,
        inputs=[inp_camera, new_camera_image, new_camera_way, new_camera_model],
        outputs=[inp_camera]
    ).then(
        pathFindingMap,
        inputs=[nodes, ways, inp_camera, inp_start, inp_goals, inp_accident_multiplier, is_show_ways, is_show_paths],
        outputs=[map,nodes,ways,inp_camera, paths_out]+camera_rows+camera_way_rows+camera_severity_rows+camera_predictions_rows+camera_image_rows
    )
    
    # Delete button event handlers
    for i, delete_btn in enumerate(camera_delete_btns):
        delete_btn.click(
            lambda cameras_df, idx=i: delete_camera(cameras_df, idx),
            inputs=[inp_camera],
            outputs=[inp_camera]
        ).then(
            pathFindingMap,
            inputs=[nodes, ways, inp_camera, inp_start, inp_goals, inp_accident_multiplier, is_show_ways, is_show_paths],
            outputs=[map,nodes,ways,inp_camera, paths_out]+camera_rows+camera_way_rows+camera_severity_rows+camera_predictions_rows+camera_image_rows
        )
    
    demo.load(pathFindingMap,[nodes, ways, inp_camera, inp_start, inp_goals, inp_accident_multiplier, is_show_ways, is_show_paths],[map,nodes,ways,inp_camera, paths_out]+camera_rows+camera_way_rows+camera_severity_rows+camera_predictions_rows+camera_image_rows)
    btn.click(pathFindingMap,[nodes, ways, inp_camera, inp_start, inp_goals, inp_accident_multiplier, is_show_ways, is_show_paths],[map,nodes,ways,inp_camera, paths_out]+camera_rows+camera_way_rows+camera_severity_rows+camera_predictions_rows+camera_image_rows)
demo.launch()