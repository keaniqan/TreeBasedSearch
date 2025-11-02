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

def _execute_with_metrics(run_fn, graph):
    """Run a search function and collect runtime and memory metrics.

    Returns: (result, runtime_seconds, peak_tracemalloc_bytes, rss_after_bytes)
    - peak_tracemalloc_bytes: peak Python allocations measured by tracemalloc
    - rss_after_bytes: process RSS at end (approx OS memory usage); None if psutil missing
    """
    # Start Python allocation tracking
    tracemalloc.start()
    t0 = time.perf_counter()
    result = run_fn(graph)
    dt = time.perf_counter() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, dt, peak


def main(filename, method, metrics_mode="none"):
    """Main function to run the search algorithm.

    metrics_mode: "none" | "stderr" | "stdout"
    - When not "none", prints a single metrics line in addition to the original output
    """
    
    # 1. Read and build the graph
    reader = GraphReader(filename)
    graph = reader.read_problem()

    # --- Print Graph for Verification (Optional) ---
    print(f"Problem File: {filename}, Method: {method}")
    print(f"Origin: {graph.origin}")
    print(f"Destinations: {graph.destinations}")
    # print(f"Nodes: {graph.nodes}")
    # print(f"Edges: {graph.adjacency}")
    
    # 2. Implement Search Logic Here
    # Choose and run the requested search method
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

    # Execute selected method with metrics
    result, runtime_s, peak_bytes = _execute_with_metrics(run_fn, graph)

    # 3. Output the result in the required format
    # Expected output:
    # <filename> <method>
    # <goal_node> <nodes_created> <path>
    goal_node, nodes_created, path_list, = result
    if result is None:
        print(f"{filename} {method}")
        print("None 0 ")
    else:
        print(f"{filename} {method}")
        print(f"{goal_node} {nodes_created}")
        print(f"{" -> ".join(str(n) for n in path_list)}")

    # Metrics (printed separately so original stdout format remains intact)
    if metrics_mode in ("stderr", "stdout"):
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
    
    # NOTE: You must implement the run_dfs, run_bfs, etc. functions separately.
    # NOTE: Remember to apply the tie-breaking rules[cite: 83, 84].

if __name__ == "__main__":
    # Check for correct command-line arguments (filename and method[, metrics flag])
    # e.g., python search.py problem.txt DFS --metrics
    if len(sys.argv) not in (3, 4):
        print("Usage: python search.py <filename> <method> [--metrics | --metrics-stdout]")
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