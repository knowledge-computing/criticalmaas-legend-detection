"""Maximum-cardinality one-to-one matches at IoU >= threshold, then maximum IoU."""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import box, iou


def match(overlaps, threshold=0.5):
    overlaps = np.asarray(overlaps, dtype=float)
    if overlaps.ndim != 2 or not overlaps.shape[0] or not overlaps.shape[1]:
        return []
    valid = overlaps >= threshold
    bonus = min(overlaps.shape) + 1  # One additional valid match outweighs every IoU improvement.
    score = np.where(valid, bonus + overlaps, 0.)
    rows, columns = linear_sum_assignment(score, maximize=True)
    return [(int(i), int(j), float(overlaps[i, j])) for i, j in zip(rows, columns) if valid[i, j]]


def summarize(tp, fp, fn, sum_iou):
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "matched_mean_iou": sum_iou / tp if tp else 0.,
            "gt_mean_iou": sum_iou / (tp + fn) if tp + fn else 0., "sum_iou": sum_iou}


def evaluate(predictions, truth, threshold=0.5, invalid_predictions=0):
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    if not isinstance(invalid_predictions, int) or invalid_predictions < 0:
        raise ValueError("Invalid-prediction count must be a nonnegative integer")
    for p in predictions + truth:
        for role in ("legend_item", "description"):
            box(p[role])
    result = {}
    overlaps = {}
    for role in ("legend_item", "description"):
        overlaps[role] = np.array([[iou(p[role], t[role]) for t in truth] for p in predictions]).reshape(len(predictions), len(truth))
        matches = match(overlaps[role], threshold)
        tp = len(matches)
        result[role] = {**summarize(tp, len(predictions) - tp + invalid_predictions, len(truth) - tp, sum(m[2] for m in matches)),
                        "matches": matches}
    # Extra diagnostic: both boxes must match the SAME truth pair; catches swapped associations.
    matches = match(np.minimum(overlaps["legend_item"], overlaps["description"]), threshold)
    tp = len(matches)
    result["linked_pair"] = {**summarize(tp, len(predictions) - tp + invalid_predictions, len(truth) - tp, sum(m[2] for m in matches)), "matches": matches}
    return result


def aggregate(results):
    return {role: summarize(*(sum(r[role][key] for r in results) for key in ("tp", "fp", "fn", "sum_iou")))
            for role in ("legend_item", "description", "linked_pair")}
