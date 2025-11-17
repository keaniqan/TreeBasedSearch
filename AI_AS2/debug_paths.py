from visualize_original import *

nodes, ways, cameras, meta = parse_assignment_file('input.txt')
osm_nodes, graph = load_osm_graph('map.osm')

# Check way 2026 (14->15)
way2026 = [w for w in ways if w['way_id']=='2026'][0]
u, v = way2026['from'], way2026['to']

snap_u = k_nearest_graph_nodes(nodes[u]['lat'], nodes[u]['lon'], osm_nodes, graph, 15)
snap_v = k_nearest_graph_nodes(nodes[v]['lat'], nodes[v]['lon'], osm_nodes, graph, 15)

print(f'Way 2026: {u} -> {v}')
print(f'Node {u} snaps to: {snap_u[:5]}')
print(f'Node {v} snaps to: {snap_v[:5]}')
print('\nTesting all snap candidate pairs:')

best_path = None
best_len = None
for i, u_osm in enumerate(snap_u):
    for j, v_osm in enumerate(snap_v):
        p = dijkstra_path(graph, u_osm, v_osm)
        if p is None:
            continue
        plen = path_length_km(p, graph)
        print(f'  [{i:2d},{j:2d}] path length: {plen:.3f} km, {len(p)} nodes')
        if best_len is None or plen < best_len:
            best_len = plen
            best_path = p

if best_path:
    print(f'\nBEST PATH: {len(best_path)} nodes, {best_len:.3f} km')
else:
    print('\nNO PATH FOUND')
