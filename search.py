import sys
import time
import tracemalloc

# Import search strategies implemented in the `strategies` package
from strategies.dfs import run_dfs
from strategies.bfs import run_bfs
from strategies.gbfs import run_gbfs
from strategies.astar import run_astar
from strategies.dijkstra import run_dijkstra
from strategies.beam import run_beam

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

# --- 1. Define Node and Graph Classes ---

class Node:
    """Represents a node in the 2D graph."""
    def __init__(self, node_id, x, y):
        self.id = int(node_id)
        self.x = int(x)
        self.y = int(y)

    def __repr__(self):
        return f"Node {self.id}: ({self.x},{self.y})"

class Graph:
    """Represents the complete directed graph."""
    def __init__(self):
        self.nodes = {}           # {node_id: Node_object}
        self.adjacency = {}       # {from_node_id: [(to_node_id, cost), ...]}
        self.origin = None        # Origin node ID
        self.destinations = set() # Set of destination node IDs

    def add_node(self, node):
        """Adds a Node object to the graph."""
        self.nodes[node.id] = node
        # Initialize adjacency list for the new node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = []

    def add_edge(self, from_id, to_id, cost):
        """Adds a directed edge and its cost."""
        if from_id in self.adjacency:
            self.adjacency[from_id].append((to_id, cost))
        else:
            self.adjacency[from_id] = [(to_id, cost)]

    def get_coordinates(self, node_id):
        """Returns the (x, y) coordinates of a node."""
        node = self.nodes.get(node_id)
        return (node.x, node.y) if node else None

    def path_cost(self, path):
        """Calculate total cost of edges in the given path."""
        total = 0
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            # Find the cost of edge from_node -> to_node
            edge_cost = None
            for neighbor, cost in self.adjacency.get(from_node, []):
                if neighbor == to_node:
                    edge_cost = cost
                    break
            if edge_cost is None:
                return None  # Edge not found
            total += edge_cost
        return total

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

# Note: You'll need to create a text file (e.g., 'problem.txt') 
# containing the data structure you provided in the previous prompt.

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

    Returns: (result, runtime_seconds, peak_tracemalloc_bytes, rss_after_bytes)
    - peak_tracemalloc_bytes: peak Python allocations measured by tracemalloc
    - rss_after_bytes: process RSS at end (approx OS memory usage); None if psutil missing
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
    result, runtime_s, peak_bytes, rss_after = _execute_with_metrics(run_fn, graph)

    # Initialize total_cost before checking it
    total_cost = None

    # 3. Output the result in the required format
    # Expected output:
    # <filename> <method>
    # <goal_node> <nodes_created> <path>
    if result is None:
        print(f"{filename} {method}")
        print("None 0 ")
        # Metrics line (optional)
        if metrics_mode in ("stderr", "stdout"):
            metrics_line = (
                f"Metrics: method={method} nodes_expanded=0 "
                f"runtime_ms={(runtime_s*1000):.3f} peak_py_mem={_format_bytes(peak_bytes)}"
                + (f" rss_now={_format_bytes(rss_after)}" if rss_after is not None else "")
            )
            if metrics_mode == "stdout":
                print(metrics_line)
            else:
                print(metrics_line, file=sys.stderr)
        return

    # Accept either (node, nodes_created, path) or (node, nodes_created, path, total_cost)
    if isinstance(result, tuple) and len(result) == 4:
        goal_node, nodes_created, path_list, total_cost = result
    else:
        goal_node, nodes_created, path_list = result
        # Calculate total cost from path
        if path_list and len(path_list) > 1:
            total_cost = graph.path_cost(path_list)
        else:
            total_cost = 0
            
    path_str = " -> ".join(str(n) for n in path_list)
    print(f"{filename} {method}")
    print(f"Goal node reached:{goal_node}")
    print(f"Number of Nodes visited:{nodes_created}")
    print(f"{path_str}")
    if total_cost is not None:
        print(f"Total path cost:{total_cost}")

    # Metrics (printed separately so original stdout format remains intact)
    if metrics_mode in ("stderr", "stdout"):
        metrics_line = (
            f"Metrics: method={method} nodes_expanded={nodes_created} "
            f"path_cost={total_cost if total_cost is not None else 'N/A'} "
            f"runtime_ms={(runtime_s*1000):.3f} peak_py_mem={_format_bytes(peak_bytes)}"
            + (f" rss_now={_format_bytes(rss_after)}" if rss_after is not None else "")
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