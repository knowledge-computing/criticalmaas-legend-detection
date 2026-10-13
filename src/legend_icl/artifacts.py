"""Isolated attempts and verified immutable output directories."""
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from .io import read_json, write_json
from .provenance import file_hash, now


def seal(directory):
    directory = Path(directory)
    files = {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p != directory / 'bundle.json'}
    write_json(directory / 'bundle.json', {'schema_version': 1, 'created_at': now(), 'artifacts': files})


def verify(directory):
    directory = Path(directory)
    manifest = read_json(directory / 'bundle.json')
    expected = manifest['artifacts']
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p != directory / 'bundle.json'}
    if actual != set(expected):
        raise ValueError(f'Bundle file inventory changed: {directory}')
    for name, digest in expected.items():
        if file_hash(directory / name) != digest:
            raise ValueError(f'Bundle artifact checksum changed: {directory / name}')
    return manifest


@contextmanager
def attempt(output, replace=False):
    output = Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / ('.' + output.name + '.lock')
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(f'Another attempt owns output directory {output}')
    os.close(descriptor)
    staging = None
    existed = output.exists() and any(output.iterdir())
    try:
        if existed and not replace:
            raise ValueError(f'Output directory is nonempty: {output}; use verified --resume or explicit --replace')
        staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '-attempt-', dir=output.parent))
        try:
            yield staging
        except BaseException:
            seal(staging)
            failures = output.parent / (output.name + '.failed-attempts')
            failures.mkdir(exist_ok=True)
            os.replace(staging, failures / uuid.uuid4().hex)
            staging = None
            raise
        seal(staging)
        if output.exists():
            if any(output.iterdir()):
                backups = output.parent / (output.name + '.previous')
                backups.mkdir(exist_ok=True)
                os.replace(output, backups / uuid.uuid4().hex)
            else:
                output.rmdir()
        os.replace(staging, output)
        staging = None
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)


def snapshot_record(record, directory, label):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = {k: v for k, v in record.items() if k not in {'record_path', 'image', 'source_image', 'annotation', 'segmentation'}}
    snapshot['original_locations'] = {k: record.get(k) for k in ('record_path', 'source_image', 'annotation', 'segmentation')}
    snapshot['source_image'] = record['map_id']  # identity is map_id + source_image_sha256; no external read required.
    if record.get('image'):
        name = label + Path(record['image']).suffix
        shutil.copyfile(record['image'], directory / name)
        snapshot['image'] = name
    else:
        snapshot['image'] = None
    write_json(directory / (label + '.json'), snapshot)
    return directory / (label + '.json')
