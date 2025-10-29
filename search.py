import sys

# Import search strategies implemented in the `strategies` package
from strategies.common import euclidean, reconstruct_path
from strategies.dfs import run_dfs
from strategies.bfs import run_bfs
from strategies.gbfs import run_gbfs
from strategies.astar import run_astar

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

# --- Search algorithm implementations (basic, suitable for small graphs) ---

# The search algorithm implementations have been moved into separate modules under
# the `strategies` package. They are imported above so the rest of this file can
# call them exactly as before (run_dfs, run_bfs, run_gbfs, run_astar).

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

def main(filename, method):
    """Main function to run the search algorithm."""
    
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
        result = run_dfs(graph)
    elif method == "BFS":
        result = run_bfs(graph)
    elif method == "GBFS":
        result = run_gbfs(graph)
    elif method == "AS":
        result = run_astar(graph)
    elif method == "CUS1":
        # Custom strategy 1 - use A*
        result = run_astar(graph)
    elif method == "CUS2":
        # Custom strategy 2 - use Greedy Best-First
        result = run_gbfs(graph)
    else:
        print(f"Unknown method: {method}")
        return

    # 3. Output the result in the required format
    # Expected output:
    # <filename> <method>
    # <goal_node> <nodes_created> <path>
    if result is None:
        print(f"{filename} {method}")
        print("None 0 ")
        return

    goal_node, nodes_created, path_list = result
    path_str = " -> ".join(str(n) for n in path_list)
    print(f"{filename} {method}")
    print(f"{goal_node} {nodes_created}\n{path_str}")
    
    # NOTE: You must implement the run_dfs, run_bfs, etc. functions separately.
    # NOTE: Remember to apply the tie-breaking rules[cite: 83, 84].

if __name__ == "__main__":
    # Check for correct command-line arguments (filename and method)
    # e.g., python search.py problem.txt DFS
    if len(sys.argv) != 3:
        # Example for expected usage (adjust 'search.py' if you rename the file)
        print("Usage: python search.py <filename> <method>") 
        print("Methods: DFS, BFS, GBFS, AS, CUS1, CUS2")
        sys.exit(1)
        
    main(sys.argv[1], sys.argv[2])