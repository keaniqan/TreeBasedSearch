import sys
import time
import tracemalloc

# Import search strategies implemented in the `strategies` package
from util import Graph, GraphReader, FormatBytes
from strategies.dfs import run_dfs
from strategies.bfs import run_bfs
from strategies.gbfs import run_gbfs
from strategies.astar import run_astar
from strategies.dijkstra import run_dijkstra
from strategies.beam import run_beam

def tsp_top_down_dp_with_memoization(graph,search_fn):
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

            # run a single-target search on a query graph to avoid mutating the main graph
            qg = _make_query_graph(current, next_node)
            result = search_fn(qg)

            if result is None:
                continue
            goal_node, n_nodes_visited, path = result
            next_path, next_cost = dp(next_node, visited.union({next_node}))

            total_cost = next_cost + graph.path_cost(path)
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


def main(filename, method, metrics_mode="none"):
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
    elif method == "DIJKSTRA":
        run_fn = run_dijkstra
    elif method == "BEAM":
        run_fn = run_beam
    else:
        print(f"Unknown method: {method}")
        return

    # Setting up metrics variables if metrics_mode is set
    t_start = None
    if metrics_mode in ("stderr", "stdout"):
        tracemalloc.start()
        t_start = time.perf_counter()

    path_list, cost=tsp_top_down_dp_with_memoization(graph, run_fn)
    
    if(path_list is None):
        print("No possible path found")
    else:
        print(f"{" -> ".join(str(n) for n in path_list)}")

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
    if len(sys.argv) not in (3, 4):
        print("Usage: python multi_search.py <filename> <method> [--metrics | --metrics-stdout]")
        print("Methods: DFS, BFS, GBFS, AS, CUS1, CUS2")
        sys.exit(1)

    filename, method = sys.argv[1], sys.argv[2]
    metrics_mode = "none"
    if len(sys.argv) == 4:
        flag = sys.argv[3].lower()
        if flag in ("--metrics", "-m"):
            metrics_mode = "stderr"
        elif flag == "--metrics-stdout":
            metrics_mode = "stdout"
        else:
            print(f"Warning: unknown flag '{sys.argv[3]}'. Metrics disabled.", file=sys.stderr)

    main(filename, method, metrics_mode)