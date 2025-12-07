import heapq
import itertools
from strategies.common import euclidean, reconstruct_path

def run_gbfs(nodes_df, ways_df, start, goals):
    """Greedy Best-First Search using Euclidean distance to nearest goal as heuristic, with DataFrames."""
    # Convert start and goals to int if possible
    try:
        start = int(start)
        goals = set(int(g) for g in goals)
    except Exception:
        goals = set(goals)

    if start is None or not goals:
        return None

    # Build adjacency list from ways_df
    adjacency = {}
    for _, row in ways_df.iterrows():
        from_id = int(row['from'])
        to_id = int(row['to'])
        if from_id not in adjacency:
            adjacency[from_id] = []
        adjacency[from_id].append((to_id, 1))  # GBFS ignores cost, so just use 1

    # Helper to get coordinates from nodes_df
    def get_coordinates(node_id):
        try:
            row = nodes_df.loc[int(node_id)]
            return (row['lat'], row['lon'])
        except Exception:
            return None

    # Filter out goal nodes that don't exist in the nodes_df
    goal_coords = []
    valid_goals = set()
    for g in goals:
        coords = get_coordinates(g)
        if coords is not None:
            goal_coords.append(coords)
            valid_goals.add(g)
    if not goal_coords:
        return None  # No valid goals

    came_from = {}
    visited = set()
    nodes_created = 0
    counter = itertools.count()
    heap = []

    # push start with its heuristic
    start_coord = get_coordinates(start)
    if start_coord is None:
        return None  # Start node doesn't exist

    start_h = min(euclidean(start_coord, gc) for gc in goal_coords)
    heapq.heappush(heap, (start_h, start, next(counter), start, None))  # (h, node_id, counter, node, parent)
    nodes_created += 1

    while heap:
        _, _node_id, _cnt, node, parent = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if parent is not None:
            came_from[node] = parent

        if node in valid_goals:
            path = reconstruct_path(came_from, node)
            return node, nodes_created, path

        # expand neighbors: push in ascending id order so insertion order reflects ascending ids
        for to_id, _ in sorted(adjacency.get(node, []), key=lambda x: x[0]):
            if to_id in visited:
                continue
            to_coord = get_coordinates(to_id)
            if to_coord is None:
                continue  # Skip nodes without coordinates
            h = min(euclidean(to_coord, gc) for gc in goal_coords)
            heapq.heappush(heap, (h, to_id, next(counter), to_id, node))
            nodes_created += 1  # Count every node created, even if already in frontier

    return None
