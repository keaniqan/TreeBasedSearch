from collections import deque
from strategies.common import reconstruct_path

def run_bfs(graph):
    """Breadth-First Search."""
    start = graph.origin
    goals = graph.destinations
    if start is None or not goals:
        return None

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
        for to_id, _cost in sorted(graph.adjacency.get(node, []), key=lambda x: x[0]):
            if to_id not in visited:
                visited.add(to_id)
                q.append((to_id, node))

    return None
