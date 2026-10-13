import math
from numbers import Real


def box(value, size=None):
    """Validate xyxy coordinates; reversed, boolean and nonfinite values are errors."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("A box must contain four xyxy coordinates")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in value):
        raise ValueError("Box coordinates must be finite numbers")
    b = list(map(float, value))
    if b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError(f"Box has zero area or reversed coordinates: {b}")
    if size and not (0 <= b[0] < b[2] <= size[0] and 0 <= b[1] < b[3] <= size[1]):
        raise ValueError(f"Box outside image {size}: {b}")
    return b


def union(boxes):
    if not boxes:
        raise ValueError("No boxes to combine")
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def offset(b, dx, dy):
    return [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]


def iou(a, b):
    intersect = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersect / (area_a + area_b - intersect) if intersect else 0.
