import math

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
