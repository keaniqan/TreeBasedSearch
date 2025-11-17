from visualize_original import *

nodes, ways, cameras, meta = parse_assignment_file('input.txt')
osm_nodes, graph = load_osm_graph('map.osm')

# Check some of the 2-node paths
test_ways = ['2026', '2007', '2014', '2018', '2022']

for way_id in test_ways:
    way = [w for w in ways if w['way_id'] == way_id][0]
    u, v = way['from'], way['to']
    
    print(f"\n=== Way {way_id}: {u} -> {v} ({way['road_name']}) ===")
    print(f"Assignment coords: ({nodes[u]['lat']:.6f}, {nodes[u]['lon']:.6f}) -> ({nodes[v]['lat']:.6f}, {nodes[v]['lon']:.6f})")
    
    snap_u = k_nearest_graph_nodes(nodes[u]['lat'], nodes[u]['lon'], osm_nodes, graph, 15)
    snap_v = k_nearest_graph_nodes(nodes[v]['lat'], nodes[v]['lon'], osm_nodes, graph, 15)
    
    # Find best path
    best_path = None
    best_len = None
    for u_osm in snap_u:
        for v_osm in snap_v:
            if u_osm == v_osm:
                continue
            path = dijkstra_path(graph, u_osm, v_osm)
            if path is None or len(path) < 2:
                continue
            plen = path_length_km(path, graph)
            if best_len is None or plen < best_len:
                best_len = plen
                best_path = path
    
    if best_path and len(best_path) == 2:
        print(f"Best path: {len(best_path)} nodes, {best_len:.3f} km")
        print(f"OSM nodes: {best_path[0]} -> {best_path[1]}")
        print(f"OSM coords: {osm_nodes[best_path[0]]} -> {osm_nodes[best_path[1]]}")
        
        # Check if these nodes are directly connected
        neighbors = [n for n, w in graph.get(best_path[0], [])]
        if best_path[1] in neighbors:
            print("✓ Direct edge exists in graph")
        else:
            print("✗ NOT directly connected - Dijkstra found intermediate route")
