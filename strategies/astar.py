import heapq
import itertools
from strategies.common import euclidean, reconstruct_path

def run_astar(graph):
    """Performs a search through the graph to find the shortest path from a source
    to a target node using A* star algorithm.

    Args:
        graph (Graph): The graph to search.

    Returns:
        tuple: (goal_node, nodes_visited, path, path_cost) if a path is found, else None.
    """
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
            # compute total path cost by summing edge costs along the path
            total_cost = 0
            if len(path) > 1:
                for i in range(len(path) - 1):
                    frm = path[i]
                    to = path[i + 1]
                    # find edge cost from frm -> to
                    edge_cost = None
                    for tid, c in graph.adjacency.get(frm, []):
                        if tid == to:
                            edge_cost = c
                            break
                    # if an edge is missing, treat its cost as 0 (keeps behaviour safe)
                    total_cost += edge_cost if edge_cost is not None else 0
            return node, nodes_created, path

        closed.add(node)

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
