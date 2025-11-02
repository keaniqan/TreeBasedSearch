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

    Returns: (result, runtime_seconds, peak_tracemalloc_bytes)
    - peak_tracemalloc_bytes: peak Python allocations measured by tracemalloc
    """
    # Start Python allocation tracking
    tracemalloc.start()
    t0 = time.perf_counter()
    result = run_fn(graph)
    dt = time.perf_counter() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, dt, peak


def main(filename, method, metrics_mode="none", num_runs=1):
    """Main function to run the search algorithm.

    metrics_mode: "none" | "stderr" | "stdout"
    - When not "none", prints a single metrics line in addition to the original output
    num_runs: number of times to run the algorithm for averaging (default 1)
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

    # Execute selected method multiple times for metrics averaging
    results = []
    runtimes = []
    peak_mems = []
    
    for i in range(num_runs):
        result, runtime_s, peak_bytes = _execute_with_metrics(run_fn, graph)
        if i == 0:
            # Store first result for output
            first_result = result
        results.append(result)
        runtimes.append(runtime_s)
        peak_mems.append(peak_bytes)

    # Calculate averages
    avg_runtime = sum(runtimes) / len(runtimes)
    avg_peak_mem = sum(peak_mems) / len(peak_mems)
    min_runtime = min(runtimes)
    max_runtime = max(runtimes)

    # 3. Output the result in the required format (using first run's result)
    result = first_result
    if result is None:
        print(f"{filename} {method}")
        print("None 0 ")
        path_cost = None
    else:
        goal_node, nodes_created, path_list = result
        path_cost = graph.path_cost(path_list)
        print(f"{filename} {method}")
        print(f"{goal_node} {nodes_created}")
        print(f"{" -> ".join(str(n) for n in path_list)}")

    # Metrics (printed separately so original stdout format remains intact)
    if metrics_mode in ("stderr", "stdout"):
        metrics_line = (
            f"METRICS ({num_runs} runs):\n"
            f"avg_runtime_ms ={avg_runtime*1000:.3f}\n"
            f"min_runtime_ms ={min_runtime*1000:.3f}\n"
            f"max_runtime_ms ={max_runtime*1000:.3f}\n"
            f"avg_peak_py_mem={FormatBytes(int(avg_peak_mem))}\n"
            f"path_cost      ={path_cost if path_cost is not None else 'N/A'}\n"
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
    num_runs = 1  # Default to 1 run
    
    if len(sys.argv) == 4:
        flag = sys.argv[3].lower()
        if flag in ("--metrics", "-m"):
            metrics_mode = "stderr"
            num_runs = 1000  # Run 100 times when metrics are enabled
        elif flag == "--metrics-stdout":
            metrics_mode = "stdout"
            num_runs = 1000  # Run 100 times when metrics are enabled
        else:
            print(f"Warning: unknown flag '{sys.argv[3]}'. Metrics disabled.", file=sys.stderr)

    main(filename, method, metrics_mode, num_runs)