import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from legend_icl.artifacts import verify
from legend_icl.comparison import compare
from legend_icl.experiments import experiment, recompute_report
from legend_icl.inference import OpenAIBackend, infer, load_record, prompt, sample_pairs
from legend_icl.io import read_json, resolve_path, write_json
from legend_icl.prepare import prepare
from legend_icl.prepare import run_lara
from legend_icl.smoke import SyntheticBackend, synthetic_data


def setup(tmp_path):
    manifest = synthetic_data(tmp_path / 'fixtures')
    data = read_json(manifest)
    return manifest, load_record(resolve_path(manifest, data['examples'][0])), load_record(resolve_path(manifest, data['targets'][0]))


def test_changed_count_dry_run_cannot_overwrite_completed_bundle(tmp_path):
    _, example, target = setup(tmp_path)
    out = tmp_path / 'run'
    infer(example, target, out, backend=SyntheticBackend())
    before = read_json(out / 'bundle.json')
    with pytest.raises(ValueError, match='nonempty'):
        infer(example, target, out, number=5, dry_run=True)
    assert read_json(out / 'bundle.json') == before
    verify(out)
    assert len(read_json(out / 'prompt.json')['examples from synthetic_example_legend.png']) == 15


def test_failed_replacement_preserves_completed_run(tmp_path):
    _, example, target = setup(tmp_path)
    out = tmp_path / 'run'
    infer(example, target, out, backend=SyntheticBackend())
    class FailedBackend:
        def infer(self, *args):
            raise RuntimeError('infrastructure failure')
    with pytest.raises(RuntimeError):
        infer(example, target, out, backend=FailedBackend(), replace=True)
    verify(out)
    assert read_json(out / 'predictions.json')['status'] == 'completed'
    assert len(list((tmp_path / 'run.failed-attempts').glob('*/run.json'))) == 1


def test_changed_truth_is_rejected_even_if_both_manifests_reference_it(tmp_path):
    manifest, _, target = setup(tmp_path)
    report = experiment(manifest, tmp_path / 'run', {'model': 'synthetic'}, backend=SyntheticBackend())
    folder = resolve_path(tmp_path / 'run/experiment.json', report['settings']['15']['runs'][0]['directory'])
    write_json(tmp_path / 'baseline.json', {'targets': [{'record': target['record_path'], 'predictions': str(folder / 'predictions.json')}]})
    mutated = read_json(target['record_path'])
    mutated['source_pairs'] = []
    write_json(target['record_path'], mutated)
    with pytest.raises(ValueError, match='ground-truth'):
        compare(tmp_path / 'run/experiment.json', tmp_path / 'baseline.json', tmp_path / 'comparison')


def test_relocated_result_recomputes_without_original_images(tmp_path):
    manifest, _, _ = setup(tmp_path)
    experiment(manifest, tmp_path / 'original', {'model': 'synthetic'}, backend=SyntheticBackend())
    shutil.copytree(tmp_path / 'original', tmp_path / 'moved')
    shutil.rmtree(tmp_path / 'original')
    shutil.rmtree(tmp_path / 'fixtures')
    report, _ = recompute_report(tmp_path / 'moved/experiment.json')
    assert report['settings']['15']['metrics']['linked_pair']['f1'] == 1
    folder = resolve_path(tmp_path / 'moved/experiment.json', report['settings']['15']['runs'][0]['directory'])
    record = load_record(folder / 'inputs/target.json')
    assert Path(record['image']).exists()


def test_artifact_tampering_is_detected(tmp_path):
    _, example, target = setup(tmp_path)
    infer(example, target, tmp_path / 'run', backend=SyntheticBackend())
    (tmp_path / 'run/evaluation_truth.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        verify(tmp_path / 'run')


def test_source_change_invalidates_resume(tmp_path, monkeypatch):
    _, example, target = setup(tmp_path)
    infer(example, target, tmp_path / 'run', backend=SyntheticBackend())
    monkeypatch.setattr('legend_icl.inference.source_identity', lambda: {'source_digest': 'changed-code'})
    with pytest.raises(ValueError, match='different inputs'):
        infer(example, target, tmp_path / 'run', backend=SyntheticBackend(), resume=True)


def test_zero_region_is_evaluated_as_false_negatives_not_dropped(tmp_path):
    manifest, _, target = setup(tmp_path)
    write_json(tmp_path / 'no-regions.json', {'segments': []})
    original = read_json(target['record_path'])
    zero = prepare(original['original_locations']['image'], tmp_path / 'zero',
        annotation=original['original_locations']['annotation'], segmentation=tmp_path / 'no-regions.json')
    data = read_json(manifest)
    data['targets'] = [str(zero)]
    data['expected_targets'] = [target['map_id']]
    write_json(manifest, data)
    class MustNotCall:
        def infer(self, *args):
            raise AssertionError('No API request should be made for a valid zero-region target')
    report = experiment(manifest, tmp_path / 'run', {'model': 'synthetic'}, backend=MustNotCall())
    assert report['planned_requests'] == 0
    assert report['settings']['15']['metrics']['legend_item']['fn'] == 20
    assert report['zero_region_maps'] == [target['map_id']]


@pytest.mark.parametrize('api', ['responses', 'chat_completions'])
@pytest.mark.parametrize('output_format', ['json_object', 'json_schema'])
def test_both_api_and_format_modes_with_real_sdk_mock_transport(tmp_path, api, output_format):
    import httpx
    from openai import OpenAI
    _, example, target = setup(tmp_path)
    def transport(request):
        body = json.loads(request.content)
        text = '{"predictions for synthetic_target_legend.png": []}'
        if api == 'responses':
            assert body['text']['format']['type'] == output_format
            return httpx.Response(200, json={'id': 'resp_test', 'object': 'response', 'status': 'completed', 'model': 'gpt-4o',
                'output': [{'id': 'msg', 'type': 'message', 'status': 'completed', 'role': 'assistant',
                            'content': [{'type': 'output_text', 'text': text, 'annotations': []}]}]})
        assert body['response_format']['type'] == output_format
        return httpx.Response(200, json={'id': 'chat_test', 'object': 'chat.completion', 'created': 0, 'model': 'gpt-4o',
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': text, 'refusal': None}}]})
    with OpenAI(api_key='offline-key', http_client=httpx.Client(transport=httpx.MockTransport(transport))) as client:
        backend = OpenAIBackend({'model': 'gpt-4o', 'api': api, 'output_format': output_format}, client)
        values, _ = backend.infer(example, target, prompt(example, target, sample_pairs(example, 15, 0)))
    assert values['predictions for synthetic_target_legend.png'] == []


def test_missing_lara_exports_are_infrastructure_failures(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    script = repo / 'pipelines/segmentation/run_pipeline.py'
    script.parent.mkdir(parents=True)
    script.touch()
    weights = tmp_path / 'weights'
    weights.mkdir()
    for name in ['config.yaml', 'config.json', 'model_final.pth']:
        (weights / name).touch()
    images = tmp_path / 'images'
    images.mkdir()
    (images / 'expected_map.png').touch()
    monkeypatch.setattr('legend_icl.prepare.subprocess.run', lambda *a, **k: None)
    with pytest.raises(RuntimeError, match='expected maps'):
        run_lara(repo, '/usr/bin/python3', weights, images, tmp_path / 'result', tmp_path / 'work')
    report = read_json(next((tmp_path / 'result.failed-attempts').glob('*/lara-run.json')))
    assert report['missing_exports'] == ['expected_map']
    assert report['status'] == 'failed-missing-exports'
    assert not (tmp_path / 'result').exists()
