import heapq
import itertools
from strategies.common import euclidean, reconstruct_path

def run_dijkstra(nodes_df, ways_df, start, goals):
    """
    Dijkstra's algorithm - uninformed shortest path search.
    Args:
        nodes_df: DataFrame of nodes (index: node id, columns: lat, lon, label)
        ways_df: DataFrame of ways (columns: id, from, to, name, type, base_time, accident_severity, final_time)
        start: start node id (string or int)
        goals: list or set of goal node ids (string or int)
    Returns:
        (goal_node, nodes_created, path) or None
    """
    # Convert start and goals to int if needed
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

    g_score = {start: 0}
    came_from = {}
    visited = set()
    nodes_created = 0
    counter = itertools.count()
    heap = [(0, start, next(counter))]  # (cost, node_id, counter)
    while heap:
        cost, node, _cnt = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        nodes_created += 1  # Count only when first expanded
        # Goal check
        if node in goals:
            path = reconstruct_path(came_from, node)
            return node, nodes_created, path
        for neighbor, edge_cost in sorted(adjacency.get(node, []), key=lambda x: x[0]):
            new_cost = cost + edge_cost
            # Only add if we found a better path
            if neighbor not in visited and new_cost < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = new_cost
                came_from[neighbor] = node
                heapq.heappush(heap, (new_cost, neighbor, next(counter)))
    return None