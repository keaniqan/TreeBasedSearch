from strategies.common import reconstruct_path

def run_dfs(graph):
    """Depth-First Search: returns (goal_node, nodes_created, path_list) or None."""
    start = graph.origin
    goals = graph.destinations
    if start is None or not goals:
        return None

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
        neighbors = graph.adjacency.get(node, [])
        for to_id, _cost in reversed(sorted(neighbors, key=lambda x: x[0])):
            if to_id not in visited:
                stack.append((to_id, node))

    return None
