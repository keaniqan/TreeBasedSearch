import plotly.graph_objects as go

def draw_way(fig, nodes: list[dict], way: dict):
    """Using plotlygraph to draw the way passed into the funciton

    Args:
        fig (:class:Figure instance): Plotly Figure instance
        nodes (list[dict]): List of nodes given as dictionaries {'lat':..., 'lon':..., 'label':...}
        way (dict): Dictionary representing a way to be drawn {'id':..., 'from':..., 'to':...,
            'name':..., 'type':(primary, secondary, tertiary, service), 'base_time':..., 'accident_severity':(0-4), 'final_time':...}
    """

    way_color="black"
    if way['type'] == "primary":
        way_color = "deepskyblue"
    elif way['type'] == "secondary":
        way_color = "yellow"
    elif way['type'] == "tertiary":
        way_color = "orange"
    elif way['type'] == "service":
        way_color = "slategray"
    hovertext=(f"{way['id']}: {way['name']}<br>"
                f"From: {way['from']}<br>"
                f"To: {way['to']}<br>")
    fig.add_trace(go.Scattermap(
        lat=[nodes[way['from']]['lat'], nodes[way['to']]['lat']],
        lon=[nodes[way['from']]['lon'], nodes[way['to']]['lon']],
        mode='lines',
        name=way['name'],
        line=dict(
            width=3,
            color=way_color
        ),
        text=hovertext,
        showlegend=False
    ))

    # Draw invisible trace marker along the path to display hovertext information along the line.
    #interpolate an array of nodes along the way for better hover effect
    density = 10
    lat_points = []
    lon_points = []
    for i in range(density+1):
        frac = i/density
        lat_points.append(nodes[way['from']]['lat']*(1-frac) + nodes[way['to']]['lat']*frac)
        lon_points.append(nodes[way['from']]['lon']*(1-frac) + nodes[way['to']]['lon']*frac)
    fig.add_trace(go.Scattermap(
        lat=lat_points,
        lon=lon_points,
        mode='markers',
        marker=dict(
            size=2,
            color='rgba(0,0,0,0)'  # invisible markers
        ),
        hoverinfo='text',
        text=hovertext,
        showlegend=False
    ))

def draw_path(fig, nodes: list[dict], ways:list[dict], path: list[str]):
    """Using plotlygraph to draw the path passed into the funciton

    Args:
        fig (:class:Figure instance): Plotly Figure instance
        nodes (list[dict]): List of nodes given as dictionaries {'id':..., 'lat':..., 'lon':...}
        ways (list[dict]): List of ways given as dictionaries {'id':..., 'from':..., 'to':...,
            'name':..., 'type':(primary, secondary, tertiary, service), 'base_time':..., 'accident_severity':..., 'final_time':...}
        path (list[str]): List of node ids representing the path to be drawn
    """