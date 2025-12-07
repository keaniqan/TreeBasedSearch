from strategies.common import reconstruct_path

def run_dfs(nodes_df, ways_df, start, goals):
    """Depth-First Search using DataFrames: returns (goal_node, nodes_created, path_list) or None."""
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
        adjacency[from_id].append((to_id, 1))  # DFS ignores cost, so just use 1

    stack = [(start, None)]  # (node, parent)
    came_from = {}
    visited = set()
    nodes_created = 0

    while stack:
        node, parent = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        nodes_created += 1
        if parent is not None:
            came_from[node] = parent

        if node in goals:
            path = reconstruct_path(came_from, node)
            return node, nodes_created, path

        # push neighbours in reverse order so smaller ids are expanded first
        neighbors = adjacency.get(node, [])
        for to_id, _ in reversed(sorted(neighbors, key=lambda x: x[0])):
            if to_id not in visited:
                stack.append((to_id, node))

    return None
