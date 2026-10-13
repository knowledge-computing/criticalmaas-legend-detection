"""Recover real scans embedded in LabelMe and pin a documented new dataset split."""
import base64
import io
from pathlib import Path

from PIL import Image

from .annotations import load_annotations
from .io import read_json, write_json
from .provenance import file_hash, now

Image.MAX_IMAGE_PIXELS = 500_000_000


def extract(directory, output):
    output = Path(output)
    scans = output / 'embedded-scans'
    scans.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted(Path(directory).glob('USGS*.json')):
        entry = {'map_id': path.stem, 'annotation_file': path.name, 'annotation_sha256': file_hash(path)}
        try:
            raw = read_json(path)
            pairs, size, warnings = load_annotations(path)
            if not raw.get('imageData'):
                raise ValueError('Annotation has no embedded original scan')
            payload = base64.b64decode(raw['imageData'], validate=True)
            with Image.open(io.BytesIO(payload)) as image:
                if list(image.size) != size:
                    raise ValueError('Embedded image and annotation dimensions differ')
                suffix = {'PNG': '.png', 'JPEG': '.jpg', 'TIFF': '.tif'}.get(image.format)
                if suffix is None:
                    raise ValueError('Unsupported embedded format')
                entry.update(image_size=size, image_format=image.format, pairs=len(pairs))
            image_path = scans / (path.stem + suffix)
            import hashlib
            digest = hashlib.sha256(payload).hexdigest()
            if image_path.exists() and file_hash(image_path) != digest:
                raise ValueError('Existing extracted scan has different bytes; use a new dataset directory')
            if not image_path.exists():
                image_path.write_bytes(payload)
            annotation = scans / (path.stem + '.pairs.json')
            write_json(annotation, {'image_size': size, 'pairs': pairs, 'source_annotation_sha256': entry['annotation_sha256']})
            entry.update(status='included', image=f'embedded-scans/{image_path.name}', image_sha256=digest,
                         annotation=f'embedded-scans/{annotation.name}', prepared_annotation_sha256=file_hash(annotation))
        except Exception as exc:
            entry.update(status='excluded', reason=str(exc))
        entries.append(entry)
    manifest = {'schema_version': 1, 'dataset_id': 'criticalmaas-local-labelme-v1',
                'created_at': now(), 'origin': 'Original scans embedded in local LabelMe annotations; no resizing',
                'historical_paper_split': False, 'maps': entries}
    write_json(output / 'local-dataset.json', manifest)
    return manifest


def validate_dataset(manifest_path, root=None):
    manifest = read_json(manifest_path)
    root = Path(root) if root else Path(manifest_path).parent
    for entry in manifest['maps']:
        if entry['status'] != 'included':
            if not entry.get('reason'):
                raise ValueError('Every exclusion must have a reason')
            continue
        for key, hash_key in [('image', 'image_sha256'), ('annotation', 'prepared_annotation_sha256')]:
            if file_hash(root / entry[key]) != entry[hash_key]:
                raise ValueError(f"Dataset artifact changed: {entry['map_id']} {key}")
    return manifest
