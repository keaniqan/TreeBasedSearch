import sys
import math
import heapq
import itertools
from collections import deque

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

def euclidean(a, b):
    """Euclidean distance between coordinate tuples a and b."""
    (x1, y1), (x2, y2) = a, b
    return math.hypot(x1 - x2, y1 - y2)

def reconstruct_path(came_from, current):
    """Reconstructs path (list of node ids) from came_from map."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def run_dfs(graph):
    """Depth-First Search: returns (goal_node, nodes_created, path_list) or None."""
    start = graph.origin
    goals = graph.destinations
    if start is None or not goals:
        return None

    stack = [(start, None)]  # (node, parent)
    came_from = {}
    visited = set()
    nodes_created = 0

    while stack:
        node, parent = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        nodes_created += 1
        if parent is not None:
            came_from[node] = parent

        if node in goals:
            path = reconstruct_path(came_from, node)
            return node, nodes_created, path

        # push neighbours in reverse order so smaller ids are expanded first
        neighbors = graph.adjacency.get(node, [])
        for to_id, _cost in reversed(sorted(neighbors, key=lambda x: x[0])):
            if to_id not in visited:
                stack.append((to_id, node))

    return None

def run_bfs(graph):
    """Breadth-First Search."""
    start = graph.origin
    goals = graph.destinations
    if start is None or not goals:
        return None

    q = deque([(start, None)])
    came_from = {}
    visited = {start}
    nodes_created = 0

    while q:
        node, parent = q.popleft()
        nodes_created += 1
        if parent is not None:
            came_from[node] = parent

        if node in goals:
            path = reconstruct_path(came_from, node)
            return node, nodes_created, path

        # enqueue neighbors in ascending id order
        for to_id, _cost in sorted(graph.adjacency.get(node, []), key=lambda x: x[0]):
            if to_id not in visited:
                visited.add(to_id)
                q.append((to_id, node))

    return None

def run_gbfs(graph):
    """Greedy Best-First Search using Euclidean distance to nearest goal as heuristic."""
    start = graph.origin
    goals = graph.destinations
    if start is None or not goals:
        return None

    goal_coords = [graph.get_coordinates(g) for g in goals]
    came_from = {}
    visited = set()
    nodes_created = 0
    counter = itertools.count()
    heap = []

    # push start with its heuristic
    start_coord = graph.get_coordinates(start)
    start_h = min(euclidean(start_coord, gc) for gc in goal_coords) if start_coord is not None else float('inf')
    heapq.heappush(heap, (start_h, start, next(counter), start, None))  # (h, node_id, counter, node, parent)

    while heap:
        _, _node_id, _cnt, node, parent = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        nodes_created += 1
        if parent is not None:
            came_from[node] = parent

        if node in goals:
            path = reconstruct_path(came_from, node)
            return node, nodes_created, path

        # expand neighbors: push in ascending id order so insertion order reflects ascending ids
        for to_id, _cost in sorted(graph.adjacency.get(node, []), key=lambda x: x[0]):
            if to_id in visited:
                continue
            to_coord = graph.get_coordinates(to_id)
            h = min(euclidean(to_coord, gc) for gc in goal_coords) if to_coord is not None else float('inf')
            heapq.heappush(heap, (h, to_id, next(counter), to_id, node))

    return None

def run_astar(graph):
    """A* search using Euclidean distance as heuristic."""
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
            return node, nodes_created, path

        closed.add(node)

        node_coord = graph.get_coordinates(node)
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