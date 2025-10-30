import matplotlib.pyplot as plt
import sys
import re

def parse_graph_file(filename):
    """Parse the graph information from the text file."""
    coords = {}
    edges = []
    origin = None
    destinations = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    current_section = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # section header detection (be tolerant of trailing spaces/colons/case)
        header = line.lower().rstrip(':').strip()
        if header == "nodes":
            current_section = "nodes"
            continue
        elif header == "edges":
            current_section = "edges"
            continue
        elif header == "origin":
            current_section = "origin"
            continue
        elif header == "destinations":
            current_section = "destinations"
            continue

        if current_section == "nodes":
            # Parse: "1: (4,1)" or "1: (4, 1)" and allow floats/negatives
            parts = line.split(':', 1)
            node_id = int(parts[0].strip())
            coord_str = parts[1]
            nums = re.findall(r"-?\d+\.?\d*", coord_str)
            if len(nums) >= 2:
                x = float(nums[0])
                y = float(nums[1])
                coords[node_id] = (x, y)
        elif current_section == "edges":
            # Parse: "(2,1): 4" (weights may be ints)
            parts = line.split(':', 1)
            edge_part = parts[0]
            weight_part = parts[1] if len(parts) > 1 else ''
            ends = re.findall(r"-?\d+", edge_part)
            if len(ends) >= 2:
                start = int(ends[0])
                end = int(ends[1])
                w_nums = re.findall(r"-?\d+\.?\d*", weight_part)
                weight = float(w_nums[0]) if w_nums else 0.0
                # store integer weights as int for prettier labels when possible
                if weight.is_integer():
                    weight = int(weight)
                edges.append((start, end, weight))
        elif current_section == "origin":
            num = re.findall(r"-?\d+", line)
            if num:
                origin = int(num[0])
        elif current_section == "destinations":
            nums = re.findall(r"-?\d+", line)
            destinations = [int(n) for n in nums]
    
    return coords, edges, origin, destinations

# Get filename from command line argument or use default
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'PathFinder-test.txt'
    print(f"No file specified, using default: {filename}")

# Read graph data from file
coords, edges, origin, destinations = parse_graph_file(filename)

# Create larger figure with better styling
plt.figure(figsize=(10, 8))
plt.style.use('default')

# Plot nodes
for node, (x, y) in coords.items():
    plt.scatter(x, y, s=500, zorder=3, color='lightblue', edgecolor='darkblue', linewidth=2)

    # Highlight origin + destination
    if node == origin:
        plt.scatter(x, y, s=600, color='lightgreen', edgecolor='darkgreen', linewidth=3, zorder=4)
    if node in destinations:
        plt.scatter(x, y, s=600, facecolors='lightcoral', edgecolor='darkred', linewidth=3, zorder=4)

# Draw all node numbers on top (after all circles are drawn)
for node, (x, y) in coords.items():
    plt.text(x, y, f"{node}", fontsize=16, ha='center', va='center', color="black", fontweight='bold', zorder=5)

# Identify bidirectional edges and build edge weight map
edge_weight_map = {}  # (start, end) -> weight
for start, end, w in edges:
    edge_weight_map[(start, end)] = w

bidirectional = set()
edges_drawn = set()  # Track which edges we've already drawn

for start, end, w in edges:
    if (end, start) in edge_weight_map:
        # Store as sorted tuple to avoid duplicates
        bidirectional.add(tuple(sorted([start, end])))

# Plot edges with arrows only for unidirectional edges
for start, end, w in edges:
    # Skip if we already drew this bidirectional edge
    edge_tuple = tuple(sorted([start, end]))
    if edge_tuple in edges_drawn:
        continue
    
    x1, y1 = coords[start]
    x2, y2 = coords[end]
    
    # Check if this edge is bidirectional
    is_bidirectional = edge_tuple in bidirectional
    
    if is_bidirectional:
        # Draw simple line for bidirectional edges
        plt.plot([x1, x2], [y1, y2], 'gray', linewidth=2, zorder=1, alpha=0.6)
        edges_drawn.add(edge_tuple)
        
        # Get both weights
        weight1 = edge_weight_map.get((start, end), w)
        weight2 = edge_weight_map.get((end, start), w)
        
        # Calculate offset perpendicular to the edge
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5
        if length > 0:
            # Perpendicular offset (normalized and scaled)
            offset_x = -dy / length * 0.3
            offset_y = dx / length * 0.3
            
            # Position weights on opposite sides of the line
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Determine which nodes for labeling
            node1, node2 = sorted([start, end])
            
            # Weight for start->end direction with arrow label
            label1 = f"{start}→{end}: {weight1}"
            plt.text(mid_x + offset_x, mid_y + offset_y, label1, 
                    fontsize=10, color='darkblue', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='blue', alpha=0.9),
                    fontweight='bold', zorder=2)
            
            # Weight for end->start direction with arrow label
            label2 = f"{end}→{start}: {weight2}"
            plt.text(mid_x - offset_x, mid_y - offset_y, label2, 
                    fontsize=10, color='darkblue', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='blue', alpha=0.9),
                    fontweight='bold', zorder=2)
    else:
        # Draw arrow for unidirectional edges
        plt.arrow(x1, y1, (x2-x1)*0.85, (y2-y1)*0.85, 
                  length_includes_head=True, head_width=0.2, head_length=0.15,
                  linewidth=2, color='darkgray', zorder=1, alpha=0.7)
        
        # Draw weight label at mid-point for unidirectional edges
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        plt.text(mid_x, mid_y, str(w), fontsize=12, color='blue', 
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue', alpha=0.8),
                 fontweight='bold', zorder=2)

plt.title("Graph Visualization", fontsize=18, fontweight='bold', pad=20)
plt.axis('equal')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

plt.show()
