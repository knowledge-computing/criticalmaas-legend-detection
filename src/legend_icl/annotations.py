"""Explicit pairs or LabelMe legend/description rectangles linked by group_id."""
from collections import defaultdict
from pathlib import Path

from .geometry import box, union
from .io import read_json, write_json

ROLES = {"legend": "legend_item", "legend_item": "legend_item", "description": "description"}


def load_annotations(path, merge_groups=False):
    raw = read_json(path)
    size = None
    warnings = []
    if isinstance(raw, dict):
        if raw.get("imageWidth") and raw.get("imageHeight"):
            size = [raw["imageWidth"], raw["imageHeight"]]
        elif raw.get("image_size"):
            size = raw["image_size"]
    if isinstance(raw, list) or (isinstance(raw, dict) and "pairs" in raw):
        pairs = raw if isinstance(raw, list) else raw["pairs"]
        result = []
        for index, p in enumerate(pairs):
            result.append({"legend_item": box(p["legend_item"], size),
                           "description": box(p["description"], size),
                           "pair_id": str(p.get("pair_id", index))})
        return result, size, warnings
    if not isinstance(raw, dict) or "shapes" not in raw:
        raise ValueError("Expected explicit pairs or LabelMe shapes")
    groups = defaultdict(lambda: defaultdict(list))
    for index, shape in enumerate(raw["shapes"]):
        role = ROLES.get(shape.get("label", "").strip().lower())
        if role is None:
            raise ValueError(f"Shape {index}: label {shape.get('label')!r} does not identify item/description; legacy labels need explicit conversion")
        group = shape.get("group_id")
        if group is None:
            raise ValueError(f"Shape {index}: group_id is missing; pairing cannot be inferred safely")
        if shape.get("shape_type") != "rectangle" or len(shape.get("points", [])) != 2:
            raise ValueError(f"Shape {index}: expected a rectangle with two corners")
        (x1, y1), (x2, y2) = shape["points"]
        groups[str(group)][role].append(box([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], size))
    result = []
    for group, roles in groups.items():
        if set(roles) != {"legend_item", "description"}:
            raise ValueError(f"Group {group} is missing its item or description")
        if any(len(v) != 1 for v in roles.values()):
            if not merge_groups:
                raise ValueError(f"Group {group} has multiple rectangles; review it or explicitly enable --merge-groups")
            warnings.append(f"Group {group}: merged multiple rectangles into bounding envelopes")
        result.append({"legend_item": union(roles["legend_item"]),
                       "description": union(roles["description"]), "pair_id": group})
    return result, size, warnings


def audit(directory, output):
    records = []
    for path in sorted(Path(directory).glob("*.json")):
        record = {"file": path.name}
        try:
            pairs, size, warnings = load_annotations(path)
            record.update(status="ready", pairs=len(pairs), image_size=size, warnings=warnings)
        except (ValueError, KeyError, TypeError) as exc:
            record.update(status="needs_review", error=str(exc))
        records.append(record)
    result = {"files": len(records), "ready": sum(r["status"] == "ready" for r in records), "records": records}
    write_json(output, result)
    return result
