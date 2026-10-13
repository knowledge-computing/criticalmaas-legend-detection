"""Synthetic offline artifacts exercise cropping, prompting, inference and metrics."""
from pathlib import Path

from PIL import Image, ImageDraw

from .artifacts import attempt
from .experiments import experiment
from .io import read_json, write_json
from .prepare import prepare


def synthetic_data(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = []
    for name in ["synthetic_example", "synthetic_target"]:
        image = Image.new("RGB", (400, 700), "white")
        draw = ImageDraw.Draw(image)
        pairs = []
        for index in range(20):
            y = 30 + index * 30
            item, description = [30, y, 50, y + 16], [70, y, 250, y + 16]
            draw.rectangle(tuple(item), fill="gray")
            draw.text((70, y), f"Synthetic unit {index}", fill="black")
            pairs.append({"legend_item": item, "description": description, "pair_id": str(index)})
        draw.text((300, 680), name, fill="black")  # Distinct source scans; prevents hash-based split leakage.
        image.save(output / f"{name}.png")
        write_json(output / f"{name}.json", {"image_size": [400, 700], "pairs": pairs})
        # Simulated LARA export tests the upstream xywh coordinate convention.
        write_json(output / f"{name}.segmentation.json", {"segments": [{"class_label": "legend_polygons",
                   "confidence": 0.9, "bbox": [20, 20, 260, 620]}]})
        targets.append(prepare(output / f"{name}.png", output / name,
            annotation=output / f"{name}.json", segmentation=output / f"{name}.segmentation.json"))
    manifest = {"examples": [str(targets[0].relative_to(output))], "targets": [str(targets[1].relative_to(output))],
                "purpose": "synthetic offline integration check, NOT the paper dataset"}
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"


class SyntheticBackend:
    """Known synthetic boxes only; this backend never calls a model."""
    def infer(self, example, target, value, normalized=False):
        pairs = [{"legend_item": [10, 10 + i * 30, 30, 26 + i * 30],
                  "description": [50, 10 + i * 30, 230, 26 + i * 30]} for i in range(20)]
        if normalized:
            pairs = [{role: [v / target["image_size"][j % 2] for j, v in enumerate(p[role])]
                      for role in ("legend_item", "description")} for p in pairs]
        return {f"predictions for {target['map_id']}_legend.png": pairs}, {"model": "synthetic-fixture", "id": "offline"}


def smoke(output):
    output = Path(output)
    with attempt(output) as staging:
        manifest = synthetic_data(staging / 'fixtures')
        result = experiment(manifest, staging / 'synthetic-results', {'model': 'synthetic-fixture'}, backend=SyntheticBackend())
        write_json(staging / 'README.json', {'synthetic': True, 'live_api_calls': 0,
                   'purpose': 'Engineering validation only. Scores are not paper results.'})
        for setting in result['settings'].values():
            assert setting['metrics']['linked_pair']['f1'] == 1.
    return {'status': 'passed', 'synthetic': True, 'live_api_calls': 0, 'counts': result['counts'],
            'output': str(output.resolve())}
