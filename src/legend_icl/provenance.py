"""Portable, canonical identities for inputs, code, requests and evaluation truth."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def environment():
    packages = {}
    for name in ('Pillow', 'openai', 'PyYAML', 'numpy', 'scipy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}


def source_identity():
    folder = Path(__file__).parent
    sources = {p.name: file_hash(p) for p in sorted(folder.glob('*.py'))}
    root = folder.parent.parent
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return {'revision': revision.stdout.strip() if revision.returncode == 0 else None,
            'source_digest': canonical_hash(sources), 'source_files': sources}


def truth_identity(record):
    return {k: record[k] for k in ('map_id', 'source_image_size', 'source_pairs', 'has_ground_truth')}


def semantic_record(record):
    # Machine locations and acquisition timestamps are informational, not coordinate semantics.
    keys = ('map_id', 'image_size', 'source_image_size', 'crop_bbox', 'crop_method',
            'coordinate_frame', 'pairs', 'source_pairs', 'has_ground_truth', 'preparation_status',
            'source_image_sha256', 'annotation_sha256', 'min_confidence', 'merge_groups')
    return {k: record.get(k) for k in keys}
