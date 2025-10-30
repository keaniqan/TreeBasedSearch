import heapq
import itertools
from strategies.common import euclidean, reconstruct_path

def run_astar(graph):
    """A* search using Euclidean distance as heuristic."""
    start = graph.origin
    goals = graph.destinations
    if start is None or not goals:
        return None

    goal_coords = [graph.get_coordinates(g) for g in goals]

    g_score = {start: 0}
    came_from = {}
    nodes_created = 0
    closed = set()
    counter = itertools.count()
    heap = []

    start_coord = graph.get_coordinates(start)
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

        node_coord = graph.get_coordinates(node)
        for to_id, cost in sorted(graph.adjacency.get(node, []), key=lambda x: x[0]):
            tentative_g = g_score.get(node, float('inf')) + cost
            if tentative_g < g_score.get(to_id, float('inf')):
                g_score[to_id] = tentative_g
                came_from[to_id] = node
                to_coord = graph.get_coordinates(to_id)
                h = min(euclidean(to_coord, gc) for gc in goal_coords) if to_coord is not None else float('inf')
                f_score = tentative_g + h
                heapq.heappush(heap, (f_score, to_id, next(counter), to_id))

    return None
