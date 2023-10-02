import numpy as np

def distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def is_within_bounds(x, y, world_map):
    """Check if the given coordinates are within the world's boundaries."""
    return 0 <= x < len(world_map[0]) and 0 <= y < len(world_map)

def is_within_radius(x1, y1, x2, y2, radius):
    """Check if the distance between two points is within the specified radius."""
    return abs(x1 - x2) <= radius and abs(y1 - y2) <= radius