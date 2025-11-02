import sys
import time
import tracemalloc
import itertools

# Import search strategies implemented in the `strategies` package
from util import Graph, GraphReader, FormatBytes
from strategies.dfs import run_dfs
from strategies.bfs import run_bfs
from strategies.gbfs import run_gbfs
from strategies.astar import run_astar
from strategies.dijkstra import run_dijkstra
from strategies.beam import run_beam

def precompute_pairwise(graph, run_fn, key_nodes=None):
    """Precompute shortest paths and costs between all ordered pairs of key nodes.

    key_nodes: iterable of node ids to consider; by default uses origin + destinations.
    Returns a dict {(a,b): {'path': path_list, 'cost': cost}, ...}
    """
    if key_nodes is None:
        key_nodes = [graph.origin] + list(graph.destinations)
    else:
        key_nodes = list(key_nodes)

    important_pairs = {}

    def _make_query_graph(src, dst):
        q = Graph()
        q.nodes = graph.nodes
        q.adjacency = graph.adjacency
        q.origin = src
        q.destinations = {dst}
        return q

    for a, b in itertools.permutations(key_nodes, 2):
        q = _make_query_graph(a, b)
        result = run_fn(q)
        if not result:
            continue
        if isinstance(result, tuple) and len(result) == 4:
            goal, _n, path, cost = result
        else:
            goal, _n, path = result
            try:
                cost = graph.path_cost(path)
            except Exception:
                continue

        if goal is None:
            continue
        important_pairs[(a, b)] = {"path": path, "cost": cost}

    return important_pairs

def tsp_top_down_dp_with_memoization(graph, search_fn, important_pairs=None):
    # Capture the full set of destinations so we don't mutate it during recursion
    all_destinations = set(graph.destinations)
    n = len(all_destinations) + 1
    memo = {}

    def _make_query_graph(src, dst):
        """Create a lightweight query graph with shared nodes/adjacency but its own origin/destinations."""
        q = Graph()
        q.nodes = graph.nodes
        q.adjacency = graph.adjacency
        q.origin = src
        q.destinations = {dst}
        return q

    def dp(current, visited):
        # Base case: all nodes visited
        if len(visited) == n:
            return [], 0

        key = (current, visited)
        if key in memo:
            return memo[key]

        best = ([], float('inf'))

        # iterate over the fixed set of destinations
        for next_node in all_destinations:
            if next_node in visited:
                continue

            # Prefer using a precomputed pairwise map when available
            entry = None
            if important_pairs is not None:
                entry = important_pairs.get((current, next_node))

            if entry is not None:
                path = entry.get("path")
                step_cost = entry.get("cost")
            else:
                # fallback to running the search function for this pair
                qg = _make_query_graph(current, next_node)
                result = search_fn(qg)
                if result is None:
                    continue
                if isinstance(result, tuple) and len(result) == 4:
                    _goal_node, _n_nodes_visited, path, step_cost = result
                else:
                    _goal_node, _n_nodes_visited, path = result
                    try:
                        step_cost = graph.path_cost(path)
                    except Exception:
                        continue

            next_path, next_cost = dp(next_node, visited.union({next_node}))

            total_cost = next_cost + (step_cost if step_cost is not None else graph.path_cost(path))
            if total_cost < best[1]:
                if next_path:
                    combined_path = path + next_path[1:]
                else:
                    combined_path = path
                best = (combined_path, total_cost)
        memo[key] = best
        return best

    # ensure visited is a frozenset for hashing
    res = dp(graph.origin, frozenset({graph.origin}))
    if res == ([], float('inf')):
        return None
    return res

def tsp_brute_force(graph, search_fn, important_pairs=None):
    """Brute-force TSP solver.

    - Computes pairwise shortest paths between the origin and all destinations using
      the provided `search_fn` (which should accept a Graph and return (goal, n, path)
      or (goal, n, path, cost)).
    - Tries every permutation of the destinations and picks the minimum-cost route.

    Returns (best_path_list, best_cost) or None if no complete tour exists.
    """
    # helper to create a query graph (single-source, single-target)
    def _make_query_graph(src, dst):
        q = Graph()
        q.nodes = graph.nodes
        q.adjacency = graph.adjacency
        q.origin = src
        q.destinations = {dst}
        return q

    # Use provided precomputed map if available, otherwise compute it now
    if important_pairs is None:
        important = precompute_pairwise(graph, search_fn)
    else:
        important = important_pairs

    destinations = list(graph.destinations)
    best_cost = float('inf')
    best_path = None

    # try all permutations of destinations
    for perm in itertools.permutations(destinations):
        cur = graph.origin
        total_cost = 0
        composed_path = [cur]
        feasible = True
        for dest in perm:
            if (cur, dest) not in important:
                feasible = False
                break
            entry = important[(cur, dest)]
            total_cost += entry['cost']
            # append path but avoid duplicating the starting node
            seg = entry['path']
            if seg:
                composed_path += seg[1:]
            cur = dest

        if feasible and total_cost < best_cost:
            best_cost = total_cost
            best_path = composed_path

    if best_path is None:
        return None
    return best_path, best_cost


def main(filename, method, metrics_mode="none", tsp_approach="DP"):
    """Main function to run the search algorithm.

    metrics_mode: "none" | "stderr" | "stdout"
    - When not "none", prints a single metrics line in addition to the original output
    """
    
    # 1. Read and build the graph
    reader = GraphReader(filename)
    graph = reader.read_problem()
    print(f"Problem File: {filename}, Method: {method}")
    print(f"Origin: {graph.origin}")
    print(f"Destinations: {graph.destinations}")
    
    # 2.Calculate shortest paths between every pair of key nodes using the search algorithm specified
    #Setting the search method to the one specified in the command line
    method = method.upper()
    if method == "DFS":
        run_fn = run_dfs
    elif method == "BFS":
        run_fn = run_bfs
    elif method == "GBFS":
        run_fn = run_gbfs
    elif method == "AS":
        run_fn = run_astar
    elif method == "CUS1":
        run_fn = run_dijkstra
    elif method == "CUS2":
        run_fn = run_beam
    else:
        print(f"Unknown method: {method}")
        return

    # Setting up metrics variables if metrics_mode is set
    t_start = None
    if metrics_mode in ("stderr", "stdout"):
        tracemalloc.start()
        t_start = time.perf_counter()

    # Precompute pairwise shortest paths between key nodes to avoid redundant searches
    important_pairs = precompute_pairwise(graph, run_fn)

    # Choose TSP solving approach: DP memoization (default) or brute-force
    tsp_key = tsp_approach.upper()
    if tsp_key == "BRUTE":
        result = tsp_brute_force(graph, run_fn, important_pairs=important_pairs)
    elif tsp_key == "DP":
        result = tsp_top_down_dp_with_memoization(graph, run_fn, important_pairs=important_pairs)
    else:
        print(f"Unknown TSP approach: {tsp_approach}")
        return

    if result is None:
        path_list = None
        cost = None
        print("No possible path found")
    else:
        path_list, cost = result
        print(" -> ".join(str(n) for n in path_list))

    #Printing out metrics if they are set
    if t_start is not None:
        runtime_s = time.perf_counter() - t_start
        _cur, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics_line = (
            "METRRICS:\n"
            f"runtime_ms ={(runtime_s*1000):.3f}\n"
            f"peak_py_mem={FormatBytes(peak_bytes)}\n"
            f"path_cost  ={graph.path_cost(path_list)}\n"
        )
        if metrics_mode == "stdout":
            print(metrics_line)
        else:
            print(metrics_line, file=sys.stderr)

if __name__ == "__main__":
    # Check for correct command-line arguments (filename and method[, metrics flag])
    # e.g., python multi_search.py problem.txt DFS --metrics
    # Accept:
    #   python multi_search.py <filename> <method> [<tsp_approach>] [--metrics | --metrics-stdout]
    if len(sys.argv) < 3:
        print("Usage: python multi_search.py <filename> <method> [<tsp_approach>] [--metrics | --metrics-stdout]")
        print("Methods: DFS, BFS, GBFS, AS, DIJKSTRA, BEAM")
        sys.exit(1)

    filename, method = sys.argv[1], sys.argv[2]
    metrics_mode = "none"
    tsp_approach = "DP"

    # parse optional args (order-flexible)
    for extra in sys.argv[3:]:
        if extra.lower() in ("--metrics", "-m"):
            metrics_mode = "stderr"
        elif extra.lower() == "--metrics-stdout":
            metrics_mode = "stdout"
        elif extra.upper() in ("BF", "BRUTE", "BRUTEFORCE", "BRUTE-FORCE", "DP", "MEMO", "DP-MEMO"):
            tsp_approach = extra
        else:
            print(f"Warning: unknown flag '{extra}'. Ignored.", file=sys.stderr)

    main(filename, method, metrics_mode, tsp_approach)