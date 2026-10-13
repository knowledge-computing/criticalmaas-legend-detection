"""Fixed map splits, isolated output bundles and offline metric recomputation."""
import csv
import hashlib
import random
import shutil
from pathlib import Path

from .artifacts import attempt, verify
from .dataset import validate_dataset
from .inference import DeferredBackend, infer, load_record, same_map, sample_pairs
from .io import read_json, resolve_path, write_json
from .metrics import aggregate, evaluate
from .provenance import canonical_hash, file_hash, semantic_record, source_identity, truth_identity


def recompute_report(report_path):
    report_path = Path(report_path)
    verify(report_path.parent)
    report = read_json(report_path)
    if report['status'] != 'completed':
        raise ValueError('Cannot report a completed experiment from failed or dry-run attempts')
    rows = []
    for count, setting in report['settings'].items():
        results = []
        for entry in setting['runs']:
            folder = resolve_path(report_path, entry['directory'])
            verify(folder)
            run = read_json(folder / 'predictions.json')
            if run['fingerprint'] != entry['fingerprint']:
                raise ValueError('Prediction/run fingerprint differs from the experiment')
            truth = read_json(resolve_path(folder / 'predictions.json', run['evaluation_truth']))
            if canonical_hash(truth) != run['evaluation_truth_sha256']:
                raise ValueError('Archived evaluation ground truth changed')
            metrics = evaluate(run['source_predictions'], truth['source_pairs'], report['threshold'], len(run['rejected']))
            results.append(metrics)
            for role, values in metrics.items():
                rows.append({'examples': int(count), 'map_id': run['target_map'], 'category': role,
                             **{k: v for k, v in values.items() if k not in {'matches', 'sum_iou'}}})
        setting['metrics'] = aggregate(results)
    return report, rows


def write_tables(output, report, rows):
    output = Path(output)
    if rows:
        with (output / 'per_map_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# Computed experiment results', '',
             'IoU: mean over matched boxes; GT-normalized IoU and all FP/FN counts are also retained in experiment.json.', '',
             '| Examples | Maps | Item IoU | Item F1 | Description IoU | Description F1 | Linked-pair F1 |',
             '|---|---|---|---|---|---|---|']
    for count, setting in report['settings'].items():
        m = setting.get('metrics')
        if m:
            lines.append(f"| {count} | {setting['maps']} | {m['legend_item']['matched_mean_iou']:.4f} | {m['legend_item']['f1']:.4f} | {m['description']['matched_mean_iou']:.4f} | {m['description']['f1']:.4f} | {m['linked_pair']['f1']:.4f} |")
    (output / 'table.md').write_text('\n'.join(lines) + '\n')


def recompute(report_path, output):
    report, rows = recompute_report(report_path)
    with attempt(output) as staging:
        write_json(staging / 'recomputed_metrics.json', {k: v['metrics'] for k, v in report['settings'].items()})
        write_tables(staging, report, rows)
    return {'status': 'completed', 'maps': len(report['target_maps']), 'output': str(output)}


def experiment(manifest_path, output, config, counts=(5, 10, 15, 20), seed=1337,
               normalized=False, dry_run=False, resume=False, backend=None, replace=False, data_root=None, protocol=None,
               run_cache=None):
    manifest = read_json(manifest_path)
    metadata_digest = None
    if manifest.get('source_metadata_manifest'):
        validate_dataset(resolve_path(manifest_path, manifest['source_metadata_manifest']),
                         data_root or resolve_path(manifest_path, manifest['dataset_root']))
        metadata_digest = file_hash(resolve_path(manifest_path, manifest['source_metadata_manifest']))
    elif protocol == 'manuscript':
        raise ValueError('Manuscript mode requires a concrete checksummed dataset manifest')
    if protocol == 'manuscript':
        if list(counts) != [5, 10, 15, 20] or normalized or not config.get('model', 'gpt-4o').startswith('gpt-4o'):
            raise ValueError('Manuscript mode requires GPT-4o, pixel coordinates and counts 5/10/15/20')
    if not counts or any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in counts) or len(set(counts)) != len(counts):
        raise ValueError('Example counts must be distinct positive integers')
    examples = [load_record(resolve_path(manifest_path, p), data_root=data_root) for p in manifest['examples']]
    targets = [load_record(resolve_path(manifest_path, p), data_root=data_root) for p in manifest['targets']]
    if not examples or not targets:
        raise ValueError('Manifest needs separate example and target records')
    if protocol == 'manuscript' and any(r.get('crop_method') != 'lara' for r in examples + targets):
        raise ValueError('Manuscript end-to-end mode requires LARA segmentation crops')
    ids = [t['map_id'] for t in targets]
    if len(set(ids)) != len(ids):
        raise ValueError('Target map IDs must be unique')
    if manifest.get('expected_targets') is not None and ids != manifest['expected_targets']:
        raise ValueError('Prepared records do not match every expected target map in order')
    if any(same_map(e, t) for e in examples for t in targets):
        raise ValueError('Example and target pools overlap: split by map before evaluation')
    if not dry_run and not all(t['has_ground_truth'] for t in targets):
        raise ValueError('Experiment evaluation needs target annotations')
    eligible = sorted([e for e in examples if len(e['pairs']) >= max(counts)], key=lambda e: e['map_id'])
    if not eligible:
        raise ValueError(f'No example legend has the required {max(counts)} visible pairs')
    experiment_fingerprint = canonical_hash({'records': [semantic_record(r) for r in examples + targets],
        'images': [file_hash(r['image']) if r.get('image') else None for r in examples + targets],
        'config': config, 'counts': counts, 'seed': seed, 'normalized': normalized,
        'source_digest': source_identity()['source_digest'], 'dataset_metadata_sha256': metadata_digest})
    output = Path(output)
    if run_cache:
        cache = Path(run_cache).resolve()
        final = output.resolve()
        if cache == final or final in cache.parents or cache in final.parents:
            raise ValueError('The run cache and final output must be separate directories')
    if resume and (output / 'experiment.json').exists():
        verify(output)
        existing = read_json(output / 'experiment.json')
        if existing.get('fingerprint') != experiment_fingerprint:
            raise ValueError('Cached experiment has different inputs/configuration or code')
        if existing['status'] != ('dry_run' if dry_run else 'completed'):
            raise ValueError('Cannot resume a different attempt type')
        return existing
    report = {'schema_version': 2, 'fingerprint': experiment_fingerprint,
              'status': 'dry_run' if dry_run else 'completed', 'seed': seed, 'counts': list(counts),
              'target_maps': ids, 'configuration': config, 'threshold': .5, 'settings': {},
              'planned_requests': len(counts) * sum(t.get('preparation_status') != 'no_legend' for t in targets),
              'zero_region_maps': [t['map_id'] for t in targets if t.get('preparation_status') == 'no_legend'],
              'protocol': manifest.get('protocol', 'documented-new-implementation'),
              'original_manifest_location': str(Path(manifest_path).resolve()),
              'source': source_identity(), 'failures': []}
    if run_cache:
        report['original_run_cache_location'] = str(Path(run_cache).resolve())
    plans = []
    for target in targets:
        target_seed = int(hashlib.sha256(f"{seed}:{target['map_id']}".encode()).hexdigest()[:16], 16)
        example = random.Random(target_seed).choice(eligible)
        selected = sample_pairs(example, max(counts), target_seed)
        plans.append((example, target, selected, target_seed))
    if not dry_run and backend is None:
        backend = DeferredBackend(config)
    with attempt(output, replace=replace) as staging:
        write_json(staging / 'dataset_manifest.json', manifest)
        write_json(staging / 'plan.json', [{'target': t['map_id'], 'example': e['map_id'], 'seed': s,
                   'pair_ids': [p['pair_id'] for p in pairs]} for e, t, pairs, s in plans])
        rows = []
        for count in counts:
            results, runs = [], []
            for example, target, selected, target_seed in plans:
                safe_id = hashlib.sha256(target['map_id'].encode()).hexdigest()[:16]
                relative_dir = f'examples_{count}/{safe_id}'
                try:
                    destination = Path(run_cache) / relative_dir if run_cache and not dry_run else staging / relative_dir
                    run = infer(example, target, destination, count, target_seed, config,
                                normalized, dry_run, backend, selected[:count], resume=bool(run_cache and not dry_run))
                    if destination != staging / relative_dir:
                        shutil.copytree(destination, staging / relative_dir)
                except Exception as exc:
                    report.update(status='failed')
                    report['failures'].append({'target': target['map_id'], 'count': count, 'kind': 'infrastructure_or_invalid_response', 'reason': str(exc)})
                    write_json(staging / 'experiment.json', report)
                    raise
                runs.append({'map_id': target['map_id'], 'directory': relative_dir, 'fingerprint': run['fingerprint']})
                if not dry_run:
                    metrics = read_json(staging / relative_dir / 'metrics.json')
                    results.append(metrics)
                    for role, values in metrics.items():
                        rows.append({'examples': count, 'map_id': target['map_id'], 'category': role,
                                     **{k: v for k, v in values.items() if k not in {'matches', 'sum_iou'}}})
            report['settings'][str(count)] = {'maps': len(runs), 'runs': runs}
            if results:
                report['settings'][str(count)]['metrics'] = aggregate(results)
        write_json(staging / 'experiment.json', report)
        if not dry_run:
            write_tables(staging, report, rows)
    return report
