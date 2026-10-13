"""Crop from LARA layout predictions; never derive target crops from ground truth."""
import math
import subprocess
from pathlib import Path

from PIL import Image

from .artifacts import attempt
from .provenance import file_hash, environment, now, source_identity, canonical_hash
from .annotations import load_annotations
from .geometry import box, offset, union
from .io import read_json, write_json

Image.MAX_IMAGE_PIXELS = 500_000_000  # The supplied USGS annotations describe ~200 MP scans.


def lara_crop(segmentation, size, min_confidence=0.25, allow_empty=False):
    if not 0 <= min_confidence <= 1:
        raise ValueError("Minimum segmentation confidence must be in [0, 1]")
    data = read_json(segmentation)
    if not isinstance(data, dict) or not isinstance(data.get('segments'), list):
        raise ValueError('Invalid LARA segmentation schema; this is not a zero-region result')
    boxes = []
    for segment in data['segments']:
        if segment.get("class_label") not in {"legend_polygons", "legend_points_lines"}:
            continue
        if segment.get("confidence", 0) < min_confidence:
            continue
        if segment.get("poly_bounds"):
            points = segment["poly_bounds"]
            boxes.append(box([min(p[0] for p in points), min(p[1] for p in points),
                              max(p[0] for p in points), max(p[1] for p in points)], size))
        else:
            x, y, width, height = segment["bbox"]  # LARA stores xywh, not xyxy.
            boxes.append(box([x, y, x + width, y + height], size))
    if not boxes:
        if allow_empty:
            return None
        raise ValueError("LARA output contains no qualifying legend region")
    return union(boxes)


def prepare(image, output, annotation=None, crop=None, segmentation=None,
            already_cropped=False, merge_groups=False, min_confidence=0.25, map_id=None, replace=False):
    if sum([crop is not None, segmentation is not None, already_cropped]) != 1:
        raise ValueError("Choose exactly one of --crop, --segmentation or --already-cropped")
    image, output = Path(image).resolve(), Path(output).resolve()
    pairs, annotated_size, warnings = load_annotations(annotation, merge_groups) if annotation else ([], None, [])
    no_legend = False
    with Image.open(image) as source:
        size = list(source.size)
        if annotated_size and list(annotated_size) != size:
            raise ValueError(f"Annotation image dimensions {annotated_size} differ from scan {size}; do not reuse resized paper figures as original scans")
        if segmentation:
            crop = lara_crop(segmentation, size, min_confidence, allow_empty=True)
            no_legend = crop is None
            if no_legend:
                crop = [0, 0, *size]
        elif already_cropped:
            crop = [0, 0, *size]
        crop = box(crop, size)
        crop = [math.floor(crop[0]), math.floor(crop[1]), math.ceil(crop[2]), math.ceil(crop[3])]
        # Validate source boxes even for explicit annotations without image dimensions.
        for pair in pairs:
            for role in ("legend_item", "description"):
                box(pair[role], size)
        cropped = None if no_legend else source.crop(tuple(crop)).convert('RGB')
    visible = []
    for pair in ([] if no_legend else pairs):
        shifted = {role: offset(pair[role], -crop[0], -crop[1]) for role in ("legend_item", "description")}
        try:
            for value in shifted.values():
                box(value, cropped.size)
        except ValueError:
            continue
        visible.append({**shifted, "pair_id": pair["pair_id"]})
    record = {"schema_version": 1, "map_id": map_id or image.stem, "image": "legend.png" if cropped else None,
              "source_image": image.name, "image_size": list(cropped.size) if cropped else size, "source_image_size": size,
              "crop_bbox": crop, "crop_method": "lara" if segmentation else "already_cropped" if already_cropped else "manual",
              "coordinate_frame": "crop_pixels", "pairs": visible, "source_pairs": pairs,
              "has_ground_truth": annotation is not None, "pairs_excluded_by_crop": len(pairs) - len(visible),
              "annotation": Path(annotation).name if annotation else None,
              "original_locations": {'image': str(image), 'annotation': str(annotation)},
              "source_image_sha256": file_hash(image),
              "annotation_sha256": file_hash(annotation) if annotation else None,
              "min_confidence": min_confidence, "merge_groups": merge_groups,
              "preparation_status": 'no_legend' if no_legend else 'prepared',
              "prepared_at": now(), "environment": environment(), "preparation_source": source_identity(),
              "warnings": warnings}
    if segmentation:
        record['segmentation'] = Path(segmentation).name
        record['segmentation_sha256'] = file_hash(segmentation)
        lara_manifest = Path(segmentation).parent / 'lara-run.json'
        if lara_manifest.exists():
            record['lara_provenance'] = read_json(lara_manifest)
    record['preparation_fingerprint'] = canonical_hash({'image': record['source_image_sha256'], 'annotation': record['annotation_sha256'],
        'crop': crop, 'method': record['crop_method'], 'confidence': min_confidence, 'merge': merge_groups,
        'segmentation': record.get('segmentation_sha256'), 'code': record['preparation_source']['source_digest']})
    with attempt(output, replace=replace) as staging:
        if cropped:
            cropped.save(staging / 'legend.png')
        write_json(staging / 'record.json', record)
    return output / 'record.json'


def run_lara(repo, python, model, images, output, workdir, cpu=False, revision=None):
    repo = Path(repo).resolve()
    if not (repo / "pipelines/segmentation/run_pipeline.py").is_file():
        raise ValueError("--repo must point to the uncharted-ta1 checkout")
    # Resolving the interpreter symlink would discard the external virtualenv.
    interpreter = str(Path(python).expanduser().absolute())
    import subprocess as sp
    current = sp.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True) if (repo / '.git').exists() else None
    commit = current.stdout.strip() if current and current.returncode == 0 else None
    if revision and revision != commit:
        raise ValueError('LARA checkout does not match the required pinned revision')
    required = ['config.yaml', 'config.json', 'model_final.pth']
    if any(not (Path(model) / name).is_file() for name in required):
        raise ValueError('LARA model directory must contain config.yaml, config.json and model_final.pth')
    model_files = {name: file_hash(Path(model) / name) for name in required}
    command = [interpreter, "-m", "pipelines.segmentation.run_pipeline",
               "--input", str(Path(images).resolve()), "--output", str(Path(output).resolve()),
               "--workdir", str(Path(workdir).resolve()), "--model", str(Path(model).resolve())]
    if cpu:
        command.append("--no_gpu")
    expected = [p.stem for p in Path(images).iterdir() if p.suffix.lower() in {'.png', '.tif', '.tiff', '.jpg', '.jpeg'}]
    with attempt(output) as staging:
        command[command.index('--output') + 1] = str(staging)
        run_manifest = {'command': command, 'requested_output': str(Path(output).absolute()),
                        'upstream_revision': commit, 'model_files': model_files, 'expected_maps': expected,
                        'started_at': now(), 'wrapper_environment': environment(), 'status': 'running'}
        if (repo / '.git').exists():
            probe = 'import sys,json,importlib.metadata as m; result={"python":sys.version}; names=["torch","detectron2","transformers","timm"];\nfor name in names:\n try: result[name]=m.version(name)\n except m.PackageNotFoundError: result[name]=None\nprint(json.dumps(result))'
            queried = subprocess.run([interpreter, '-c', probe], capture_output=True, text=True)
            if queried.returncode == 0:
                import json
                run_manifest['external_environment'] = json.loads(queried.stdout)
            else:
                raise RuntimeError('Cannot identify the external LARA environment')
        write_json(staging / 'lara-run.json', run_manifest)
        try:
            subprocess.run(command, cwd=repo, check=True)
            missing = [name for name in expected if not (staging / (name + '_map_segmentation.json')).exists()]
            run_manifest['missing_exports'] = missing
            run_manifest['status'] = 'completed' if not missing else 'failed-missing-exports'
            if missing:
                raise RuntimeError(f'LARA did not export segmentation for {len(missing)} expected maps: {missing}')
        except BaseException:
            if run_manifest['status'] == 'running':
                run_manifest['status'] = 'failed'
            raise
        finally:
            run_manifest['finished_at'] = now()
            write_json(staging / 'lara-run.json', run_manifest)
