import heapq
import itertools
from strategies.common import euclidean, reconstruct_path

def run_dijkstra(graph):
    """Dijkstra's algorithm - uninformed shortest path search."""
    start = graph.origin
    goals = graph.destinations
    if start is None or not goals:
        return None
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
            return node, nodes_created, path, cost
        for neighbor, edge_cost in sorted(graph.adjacency.get(node, []), key=lambda x: x[0]):
            new_cost = cost + edge_cost
            # Only add if we found a better path
            if neighbor not in visited and new_cost < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = new_cost
                came_from[neighbor] = node
                heapq.heappush(heap, (new_cost, neighbor, next(counter)))
    return None