import plotly.graph_objects as go

def draw_way(fig, nodes, way):
    way_color="black";
    if way['type'] == "primary":
        way_color = "deepskyblue"
    elif way['type'] == "secondary":
        way_color = "yellow"
    elif way['type'] == "tertiary":
        way_color = "orange"
    elif way['type'] == "service":
        way_color = "slategray"
    
    fig.add_trace(go.Scattermap(
        lat=[nodes[way['from']]['lat'], nodes[way['to']]['lat']],
        lon=[nodes[way['from']]['lon'], nodes[way['to']]['lon']],
        mode='lines',

        #Defining colors of the ways. Depending on the highway 
        line=go.scattermap.Line(
            width=2,
            color = way_color
        ),
        hoverinfo='none'
    ))