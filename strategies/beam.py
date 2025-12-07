from collections import deque
import itertools

from strategies.common import euclidean, reconstruct_path


def run_beam(nodes_df, ways_df, start, goals, beam_width=2):
    """
    Beam search over the graph using DataFrames.
    Args:
        nodes_df: DataFrame of nodes (index: node id, columns: lat, lon, label)
        ways_df: DataFrame of ways (columns: id, from, to, name, type, base_time, accident_severity, final_time)
        start: start node id (string or int)
        goals: list or set of goal node ids (string or int)
        beam_width: maximum number of nodes to keep at each level
    Returns:
        (goal_node, nodes_visited, path_list) if a goal is found, else None
    """
    # Convert start and goals to int if possible
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

    # Helper to get coordinates from nodes_df
    def get_coordinates(node_id):
        try:
            row = nodes_df.loc[int(node_id)]
            return (row['lat'], row['lon'])
        except Exception:
            return None

    visited = set([start])
    came_from = {}
    nodes_visited = 0
    if start is None or not goals:
        return None
    goal_coords = [get_coordinates(g) for g in goals]

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
            for to_id, _cost in sorted(adjacency.get(node, []), key=lambda x: x[0]):
                if to_id in visited:
                    continue
                to_coord = get_coordinates(to_id)
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
            came_from[to_id] = parent
            visited.add(to_id)
            nodes_visited += 1
            next_beam.append(to_id)
            if len(next_beam) >= beam_width:
                break
        current = next_beam

    return None