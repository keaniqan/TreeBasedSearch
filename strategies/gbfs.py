import heapq
import itertools
from strategies.common import euclidean, reconstruct_path

def run_gbfs(graph):
    """Greedy Best-First Search using Euclidean distance to nearest goal as heuristic."""
    start = graph.origin
    goals = graph.destinations
    if start is None or not goals:
        return None

    goal_coords = [graph.get_coordinates(g) for g in goals]
    came_from = {}
    visited = set()
    nodes_created = 0
    counter = itertools.count()
    heap = []

    # push start with its heuristic
    start_coord = graph.get_coordinates(start)
    start_h = min(euclidean(start_coord, gc) for gc in goal_coords) if start_coord is not None else float('inf')
    heapq.heappush(heap, (start_h, start, next(counter), start, None))  # (h, node_id, counter, node, parent)
    nodes_created += 1

    while heap:
        _, _node_id, _cnt, node, parent = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if parent is not None:
            came_from[node] = parent

        if node in goals:
            path = reconstruct_path(came_from, node)
            return node, nodes_created, path

        # expand neighbors: push in ascending id order so insertion order reflects ascending ids
        for to_id, _cost in sorted(graph.adjacency.get(node, []), key=lambda x: x[0]):
            if to_id in visited:
                continue
            to_coord = graph.get_coordinates(to_id)
            h = min(euclidean(to_coord, gc) for gc in goal_coords) if to_coord is not None else float('inf')
            heapq.heappush(heap, (h, to_id, next(counter), to_id, node))
            nodes_created += 1  # Count every node created, even if already in frontier

    return None
