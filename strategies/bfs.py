from collections import deque
from strategies.common import reconstruct_path

def run_bfs(nodes_df, ways_df, start, goals):
    """Breadth-First Search using DataFrames."""
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
        adjacency[from_id].append((to_id, 1))  # BFS ignores cost, so just use 1

    q = deque([(start, None)])
    came_from = {}
    visited = {start}
    nodes_created = 0

    while q:
        node, parent = q.popleft()
        nodes_created += 1
        if parent is not None:
            came_from[node] = parent

        if node in goals:
            path = reconstruct_path(came_from, node)
            return node, nodes_created, path

        # enqueue neighbors in ascending id order
        for to_id, _ in sorted(adjacency.get(node, []), key=lambda x: x[0]):
            if to_id not in visited:
                visited.add(to_id)
                q.append((to_id, node))

    return None
