"""Compare external item-description predictions to saved in-context runs."""
import os
from pathlib import Path

from .experiments import recompute_report
from .provenance import canonical_hash, truth_identity
from .geometry import box, offset
from .inference import load_record
from .io import read_json, resolve_path, write_json
from .metrics import aggregate, evaluate


def evaluate_saved(prediction, target, threshold=0.5):
    if prediction.get("target_map") != target["map_id"]:
        raise ValueError("Prediction target_map must match the target record")
    if not target["has_ground_truth"]:
        raise ValueError("Target has no ground truth")
    if prediction.get('evaluation_truth_sha256') and prediction['evaluation_truth_sha256'] != canonical_hash(truth_identity(target)):
        raise ValueError('Prediction and target ground-truth revisions differ')
    frame = prediction.get("coordinate_frame")
    if frame not in {"crop_pixels", "source_pixels"}:
        raise ValueError("Prediction coordinate_frame must be crop_pixels or source_pixels")
    pairs = prediction["predictions"]
    if not isinstance(pairs, list):
        raise ValueError("predictions must be a list")
    for pair in pairs:
        for role in ("legend_item", "description"):
            box(pair[role], target["image_size"] if frame == "crop_pixels" else target["source_image_size"])
    if frame == "crop_pixels":
        dx, dy = target["crop_bbox"][:2]
        pairs = [{role: offset(p[role], dx, dy) for role in ("legend_item", "description")} for p in pairs]
    return evaluate(pairs, target["source_pairs"], threshold, len(prediction.get("rejected", [])))


def compare(experiment_path, baseline_manifest, output, count=15):
    report, _ = recompute_report(experiment_path)
    if report.get("status") != "completed":
        raise ValueError("Comparison requires a completed in-context experiment")
    setting = report["settings"][str(count)]
    manifest = read_json(baseline_manifest)
    entries = manifest["targets"]
    records = [load_record(resolve_path(baseline_manifest, entry["record"]), require_image=False) for entry in entries]
    ids = [r["map_id"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(report["target_maps"]):
        raise ValueError("Baseline and in-context experiment must evaluate exactly the same target maps")
    reference_records = {}
    returned_models = set()
    requested_models = set()
    for entry in setting['runs']:
        folder = resolve_path(experiment_path, entry['directory'])
        run = read_json(folder / 'predictions.json')
        frozen = read_json(resolve_path(folder / 'predictions.json', run['evaluation_truth']))
        reference_records[entry['map_id']] = frozen
        if run.get('model_returned'):
            returned_models.add(run['model_returned'])
        if run.get('model_requested'):
            requested_models.add(run['model_requested'])
    requested = report.get('configuration', {}).get('model')
    requested = requested or ', '.join(sorted(requested_models)) or 'In-context model'
    method = ', '.join(sorted(returned_models)) or requested
    baseline_method = manifest.get('method') or 'External baseline'
    per_map = []
    for entry, record in zip(entries, records):
        reference = reference_records[record['map_id']]
        if canonical_hash(truth_identity(record)) != canonical_hash(reference):
            raise ValueError("Baseline and in-context experiment ground-truth records differ")
        metrics = evaluate_saved(read_json(resolve_path(baseline_manifest, entry["predictions"])), record, report["threshold"])
        per_map.append({"map_id": record["map_id"], "metrics": metrics})
    output = Path(output)
    result = {"maps": ids, "threshold": report["threshold"], "baseline": aggregate([p["metrics"] for p in per_map]),
              "in_context": setting["metrics"],
              "gpt": setting["metrics"],  # Legacy alias retained for existing consumers.
              "model_requested": requested, "models_returned": sorted(returned_models),
              "experiment_method": method, "baseline_method": baseline_method,
              "num_examples": count, "baseline_per_map": per_map,
              "baseline_manifest": os.path.relpath(Path(baseline_manifest).resolve(), output.resolve()),
              "experiment": os.path.relpath(Path(experiment_path).resolve(), output.resolve())}
    write_json(output / "comparison.json", result)
    lines = ["| Method | Item IoU | Item F1 | Description IoU | Description F1 |",
             "|---|---|---|---|---|"]
    for label, values in [(baseline_method, result["baseline"]), (f"{method} ({count} ex.)", result["in_context"])]:
        lines.append(f"| {label} | {values['legend_item']['matched_mean_iou']:.4f} | {values['legend_item']['f1']:.4f} | {values['description']['matched_mean_iou']:.4f} | {values['description']['f1']:.4f} |")
    (output / "table.md").write_text("\n".join(lines) + "\n")
    return result
