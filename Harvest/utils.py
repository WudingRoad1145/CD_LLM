import numpy as np

def distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def is_within_bounds(x, y, world_map):
    """Check if the given coordinates are within the world's boundaries."""
    return 0 <= x < len(world_map[0]) and 0 <= y < len(world_map)

def is_within_radius(x1, y1, x2, y2, radius):
    """Check if the distance between two points is within the specified radius."""
    return abs(x1 - x2) <= radius and abs(y1 - y2) <= radius

def extract_submatrix(matrix, x, y, scope):
    # Define the boundaries for extraction
    x_start = max(0, x - scope)
    x_end = min(len(matrix), x + scope + 1)
    y_start = max(0, y - scope)
    y_end = min(len(matrix[0]), y + scope + 1)
    
    # Extract the submatrix
    submatrix = [row[y_start:y_end] for row in matrix[x_start:x_end]]
    return submatrix
