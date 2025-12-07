import heapq
import itertools
from strategies.common import euclidean, reconstruct_path

def run_astar(nodes_df, ways_df, start, goals):
    """
    Performs A* search to find the shortest path from start to any of the goal nodes.
    Args:
        nodes_df: DataFrame of nodes (index: node id, columns: lat, lon, label)
        ways_df: DataFrame of ways (columns: id, from, to, name, type, base_time, accident_severity, final_time)
        start: start node id (string or int)
        goals: list or set of goal node ids (string or int)
    Returns:
        (goal_node, nodes_created, path) or None
    """
    # Convert start and goals to int if possible
    try:
        start = int(start)
        goals = set(int(g) for g in goals)
    except Exception:
        goals = set(goals)

    # Build adjacency list from ways_df
    adjacency = {}
    for _, row in ways_df.iterrows():
        from_id = int(row['from'])
        to_id = int(row['to'])
        cost = float(row['final_time']) if 'final_time' in row else float(row['base_time'])
        if from_id not in adjacency:
            adjacency[from_id] = []
        adjacency[from_id].append((to_id, cost))

    # Helper to get coordinates from nodes_df
    def get_coordinates(node_id):
        try:
            row = nodes_df.loc[int(node_id)]
            return (row['lat'], row['lon'])
        except Exception:
            return None

    goal_coords = [get_coordinates(g) for g in goals]

    g_score = {start: 0}
    came_from = {}
    nodes_created = 0
    closed = set()
    counter = itertools.count()
    heap = []

    start_coord = get_coordinates(start)
    start_h = min(euclidean(start_coord, gc) for gc in goal_coords) if start_coord is not None else float('inf')
    heapq.heappush(heap, (start_h, start, next(counter), start))

    while heap:
        f, _node_id, _cnt, node = heapq.heappop(heap)
        if node in closed:
            continue
        nodes_created += 1

        if node in goals:
            path = reconstruct_path(came_from, node)
            return node, nodes_created, path

        closed.add(node)

        for to_id, cost in sorted(adjacency.get(node, []), key=lambda x: x[0]):
            tentative_g = g_score.get(node, float('inf')) + cost
            if tentative_g < g_score.get(to_id, float('inf')):
                g_score[to_id] = tentative_g
                came_from[to_id] = node
                to_coord = get_coordinates(to_id)
                h = min(euclidean(to_coord, gc) for gc in goal_coords) if to_coord is not None else float('inf')
                f_score = tentative_g + h
                heapq.heappush(heap, (f_score, to_id, next(counter), to_id))

    return None
