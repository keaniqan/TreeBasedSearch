import sys
import time
import tracemalloc
import itertools

# Import search strategies implemented in the `strategies` package
from strategies.astar import run_astar
from common import Graph, Node

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None



# --- 2. Implement the File Reader (Parser) ---
class GraphReader:
    """Handles parsing the problem specification file."""
    
    def __init__(self, filename):
        self.filename = filename
        self.graph = Graph()
        
    def read_problem(self):
        """Reads the file and populates the Graph object."""
        try:
            with open(self.filename, 'r') as f:
                lines = [line.strip() for line in f if line.strip()] # Read and clean lines
        except FileNotFoundError:
            print(f"Error: File not found: {self.filename}")
            sys.exit(1)

        current_section = None
        
        for line in lines:
            if line.startswith("Nodes:"):
                current_section = "NODES"
                continue
            elif line.startswith("Edges:"):
                current_section = "EDGES"
                continue
            elif line.startswith("Origin:"):
                current_section = "ORIGIN"
                continue
            elif line.startswith("Destinations:"):
                current_section = "DESTINATIONS"
                continue
            
            if current_section == "NODES":
                # Example: 1: (4,1)
                try:
                    parts = line.split(':')
                    node_id = int(parts[0].strip())
                    coords_str = parts[1].strip().strip('()')
                    x, y = map(int, coords_str.split(','))
                    self.graph.add_node(Node(node_id, x, y))
                except Exception as e:
                    print(f"Error parsing node line '{line}': {e}")
            
            elif current_section == "EDGES":
                # Example: (2,1): 4
                try:
                    parts = line.split(':')
                    cost = int(parts[1].strip())
                    
                    # Extract (2,1)
                    nodes_str = parts[0].strip().strip('()')
                    from_id, to_id = map(int, nodes_str.split(','))
                    
                    self.graph.add_edge(from_id, to_id, cost)
                except Exception as e:
                    print(f"Error parsing edge line '{line}': {e}")
            
            elif current_section == "ORIGIN":
                # Example: 2
                try:
                    self.graph.origin = int(line)
                except Exception as e:
                    print(f"Error parsing origin line '{line}': {e}")
                    
            elif current_section == "DESTINATIONS":
                # Example: 5; 4
                try:
                    dest_ids = [int(d.strip()) for d in line.split(';') if d.strip()]
                    self.graph.destinations.update(dest_ids)
                except Exception as e:
                    print(f"Error parsing destinations line '{line}': {e}")
                    
        return self.graph

# --- 3. Main Program Structure (Entry Point) ---
def _format_bytes(n_bytes: int) -> str:
    """Human-readable bytes in KB/MB with 2 decimals."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    kb = n_bytes / 1024.0
    if kb < 1024:
        return f"{kb:.2f} KB"
    mb = kb / 1024.0
    if mb < 1024:
        return f"{mb:.2f} MB"
    gb = mb / 1024.0
    return f"{gb:.2f} GB"


def _execute_with_metrics(run_fn, graph):
    """Run a search function and collect runtime and memory metrics.

    Returns: 
        tuple: (path, cost) or else None if no path found.
    """
    # Start Python allocation tracking
    tracemalloc.start()
    # Optional OS RSS sampling
    proc = psutil.Process() if psutil else None
    t0 = time.perf_counter()
    result = run_fn(graph)
    dt = time.perf_counter() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = proc.memory_info().rss if proc else None
    return result, dt, peak, rss_after

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
    
    '''1. Read and build the graph'''
    reader = GraphReader(filename)
    graph = reader.read_problem()
    print(f"Problem File: {filename}, Method: {method}")
    print(f"Origin: {graph.origin}")
    print(f"Destinations: {graph.destinations}")
    
    '''2.Calculate shortest paths between every pair of key nodes using the search algorithm specified'''
    #Setting the search method to the one specified in the command line
    match method.upper():
        case "AS":
            run_fn = run_astar
        case _:
            print(f"Unknown method: {method}")
            return

    res=tsp_top_down_dp_with_memoization(graph, run_fn)

    if res is None:
        print("No possible path found")
    else:
        print(f"Optimal TSP Path: {res[0]}")
        print(f"Total Path Cost: {res[1]}")

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