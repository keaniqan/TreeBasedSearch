from collections import deque
import itertools

from strategies.common import euclidean, reconstruct_path


def run_beam(graph, beam_width=2):
    """Beam search over the graph.

    Args:
        graph: Graph object (must provide origin, destinations, adjacency, get_coordinates)
        beam_width: maximum number of nodes to keep at each level

    Returns:
        (goal_node, nodes_visited, path_list) if a goal is found, else None
    """
    start = graph.origin
    goals = graph.destinations
    visited = set([start])
    came_from = {}
    nodes_visited = 0
    #Retrieving the list of goals and its coordinates to calculate heuristic
    if start is None or not goals:
        return None
    goal_coords = [graph.get_coordinates(g) for g in goals]
    
    # current beam (list of node ids)
    current = [start]
    counter = itertools.count()

    while current:
        # check if any current node is a goal
        for node in current:
            if node in goals:
                path = reconstruct_path(came_from, node)
                return node, nodes_visited, path

        # gather successors with heuristic scores
        successors = []  # list of (h, tie_breaker, node_id, parent)
        for node in current:
            for to_id, _cost in sorted(graph.adjacency.get(node, []), key=lambda x: x[0]):
                if to_id in visited:
                    continue
                to_coord = graph.get_coordinates(to_id)
                if to_coord is None:
                    h = float('inf')
                else:
                    h = min(euclidean(to_coord, gc) for gc in goal_coords)
                successors.append((h, to_id, next(counter), to_id, node))
        
        # If no successors left that means no path is possible
        if not successors:
            return None

        # sort successors by heuristic (then by insertion order) and pick top-k unique nodes
        successors.sort(key=lambda x: (x[0], x[2]))
        next_beam = []
        seen = set()
        for item in successors:
            _, to_id, _cnt, _node, parent = item
            if to_id in seen:
                continue
            seen.add(to_id)
            # register parent mapping now so reconstruct_path can be used later
            came_from[to_id] = parent
            visited.add(to_id)
            nodes_visited += 1
            next_beam.append(to_id)
            if len(next_beam) >= beam_width:
                break
        current = next_beam

    return None