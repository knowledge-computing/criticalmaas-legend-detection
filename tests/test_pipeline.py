import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from legend_icl.annotations import audit, load_annotations
from legend_icl.comparison import compare, evaluate_saved
from legend_icl.experiments import experiment
from legend_icl.geometry import box, iou
from legend_icl.inference import OpenAIBackend, decode_predictions, infer, load_record, prompt, sample_pairs
from legend_icl.io import read_json, write_json, resolve_path
from legend_icl.metrics import evaluate, match
from legend_icl.prepare import lara_crop, prepare, run_lara
from legend_icl.smoke import SyntheticBackend, synthetic_data


def pair(x=0):
    return {"legend_item": [x, 0, x + 10, 10], "description": [x, 20, x + 10, 30]}


@pytest.mark.parametrize("value", [[0, 0, 0, 10], [10, 0, 0, 10], [True, 0, 10, 10], [0, 0, float('nan'), 10], [0, 0, '10', 10]])
def test_invalid_box(value):
    with pytest.raises(ValueError):
        box(value)


def test_iou_and_threshold_boundary():
    assert iou([0, 0, 10, 10], [0, 0, 20, 10]) == .5
    assert match([[.5]]) == [(0, 0, .5)]


def test_matching_maximizes_cardinality_not_greedy_iou():
    assert {(i, j) for i, j, _ in match([[.9, .8], [.85, 0]])} == {(0, 1), (1, 0)}


def test_duplicate_predictions_and_invalid_outputs_are_false_positives():
    result = evaluate([pair(), pair()], [pair()], invalid_predictions=1)
    assert result['legend_item']['tp'] == 1
    assert result['legend_item']['fp'] == 2
    assert result['legend_item']['f1'] == .5


def test_swapped_links_cannot_pass_pair_metric():
    a, b = pair(), pair(40)
    predictions = [{"legend_item": a['legend_item'], "description": b['description']},
                   {"legend_item": b['legend_item'], "description": a['description']}]
    metrics = evaluate(predictions, [a, b])
    assert metrics['legend_item']['f1'] == metrics['description']['f1'] == 1
    assert metrics['linked_pair']['f1'] == 0


def test_empty_predictions_and_empty_truth():
    assert evaluate([], [pair()])['legend_item']['fn'] == 1
    assert evaluate([pair()], [])['legend_item']['fp'] == 1
    assert evaluate([], [])['linked_pair']['f1'] == 0


def test_labelme_groups_do_not_depend_on_shape_order(tmp_path):
    annotation = tmp_path / 'ann.json'
    write_json(annotation, {"shapes": [
        {"label": "description", "group_id": 3, "shape_type": "rectangle", "points": [[50, 60], [20, 30]]},
        {"label": "legend", "group_id": 3, "shape_type": "rectangle", "points": [[0, 0], [10, 10]]}]})
    pairs, _, _ = load_annotations(annotation)
    assert pairs == [{"pair_id": "3", "legend_item": [0., 0., 10., 10.], "description": [20., 30., 50., 60.]}]
    data = read_json(annotation)
    data['shapes'].append({"label": "description", "group_id": 3, "shape_type": "rectangle", "points": [[60, 60], [70, 70]]})
    write_json(annotation, data)
    with pytest.raises(ValueError, match="multiple rectangles"):
        load_annotations(annotation)
    pairs, _, warnings = load_annotations(annotation, merge_groups=True)
    assert pairs[0]['description'] == [20., 30., 70., 70.]
    assert warnings


def test_missing_group_is_not_guessed(tmp_path):
    annotation = tmp_path / 'ann.json'
    write_json(annotation, {"shapes": [{"label": "legend", "group_id": None, "shape_type": "rectangle", "points": [[0, 0], [1, 1]]}]})
    with pytest.raises(ValueError, match="group_id"):
        load_annotations(annotation)


def test_lara_xywh_and_multiple_legend_regions(tmp_path):
    data = tmp_path / 'lara.json'
    write_json(data, {"segments": [
        {"class_label": "map", "confidence": 1, "bbox": [0, 0, 100, 100]},
        {"class_label": "legend_polygons", "confidence": .8, "bbox": [10, 20, 30, 40]},
        {"class_label": "legend_points_lines", "confidence": .7, "poly_bounds": [[60, 70], [80, 90]]}]})
    assert lara_crop(data, [100, 100]) == [10, 20, 80, 90]


def test_crop_misses_count_as_false_negatives(tmp_path):
    image = tmp_path / 'map.png'
    Image.new('RGB', (100, 100)).save(image)
    annotation = tmp_path / 'ann.json'
    write_json(annotation, {"image_size": [100, 100], "pairs": [pair(), pair(60)]})
    record = load_record(prepare(image, tmp_path / 'crop', annotation=annotation, crop=[0, 0, 50, 50]))
    assert len(record['pairs']) == 1 and record['pairs_excluded_by_crop'] == 1
    assert evaluate([pair()], record['source_pairs'])['legend_item']['fn'] == 1


def test_mismatched_annotation_dimensions_are_errors(tmp_path):
    Image.new('RGB', (10, 10)).save(tmp_path / 'image.png')
    write_json(tmp_path / 'ann.json', {"image_size": [20, 20], "pairs": []})
    with pytest.raises(ValueError, match="dimensions"):
        prepare(tmp_path / 'image.png', tmp_path / 'crop', annotation=tmp_path / 'ann.json', already_cropped=True)


def test_output_normalization_is_explicit_and_rejected_boxes_are_recorded():
    target = {"map_id": "target", "image_size": [100, 200]}
    output = {"predictions for target_legend.png": [
        {"legend_item": [.1, .1, .2, .2], "description": [.3, .3, .4, .4]},
        {"legend_item": [0, 0, 0, 0], "description": [1, 1, 2, 2]}]}
    accepted, rejected = decode_predictions(output, target, normalized=True)
    assert accepted[0]['legend_item'] == [10, 20, 20, 40]
    assert len(rejected) == 1


def records(tmp_path):
    manifest_path = synthetic_data(tmp_path / 'fixtures')
    manifest = read_json(manifest_path)
    return manifest_path, load_record(resolve_path(manifest_path, manifest['examples'][0])), load_record(resolve_path(manifest_path, manifest['targets'][0]))


def test_target_annotations_are_never_in_prompt_and_k_is_exact(tmp_path):
    _, example, target = records(tmp_path)
    selected = sample_pairs(example, 15, 123)
    target['pairs'] = [{'legend_item': [999, 999, 1000, 1000], 'description': [999, 999, 1000, 1000]}]
    value = prompt(example, target, selected)
    assert len(value['examples from synthetic_example_legend.png']) == 15
    assert '999' not in json.dumps(value)
    with pytest.raises(ValueError):
        sample_pairs(example, 21, 123)


def test_same_map_leakage_is_rejected(tmp_path):
    _, example, _ = records(tmp_path)
    with pytest.raises(ValueError, match="different maps"):
        infer(example, example, tmp_path / 'run', dry_run=True)


def test_full_experiment_nested_examples_and_resume(tmp_path):
    manifest_path, example, target = records(tmp_path)
    result = experiment(manifest_path, tmp_path / 'runs', {"model": "synthetic"}, backend=SyntheticBackend())
    prompts = []
    for k in [5, 10, 15, 20]:
        entry = result['settings'][str(k)]
        assert entry['metrics']['legend_item']['f1'] == 1
        directory = resolve_path(tmp_path / 'runs/experiment.json', entry['runs'][0]['directory'])
        run = read_json(directory / 'predictions.json')
        assert run['source_predictions'][0]['legend_item'] == [30, 30, 50, 46]
        prompts.append(read_json(directory / 'prompt.json')['examples from synthetic_example_legend.png'])
    assert prompts[0] == prompts[-1][:5] and prompts[2] == prompts[-1][:15]
    experiment(manifest_path, tmp_path / 'runs', {"model": "synthetic"}, resume=True)  # No backend or API key needed.
    with pytest.raises(ValueError, match="different inputs"):
        experiment(manifest_path, tmp_path / 'runs', {"model": "changed"}, resume=True)


def test_missing_prediction_key_is_failed_not_zero_detections(tmp_path):
    _, example, target = records(tmp_path)
    class BadBackend:
        def infer(self, *args):
            return {"wrong_key": []}, {"id": "bad-response"}
    with pytest.raises(ValueError, match="missing list"):
        infer(example, target, tmp_path / 'run', backend=BadBackend())
    assert read_json(next((tmp_path / 'run.failed-attempts').glob('*/run.json')))['status'] == 'failed'
    assert not (tmp_path / 'run/predictions.json').exists()


def test_api_payload_uses_data_urls_and_responses_format(tmp_path):
    _, example, target = records(tmp_path)
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        value = {"predictions for synthetic_target_legend.png": []}
        return SimpleNamespace(output_text=json.dumps(value), status='completed',
                               model_dump=lambda **_: {"id": "offline", "model": "gpt-4o"})
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    backend = OpenAIBackend({"model": "gpt-4o", "temperature": 0}, client)
    backend.infer(example, target, prompt(example, target, sample_pairs(example, 15, 0)))
    assert captured['text']['format']['type'] == 'json_object'
    assert 'response_format' not in captured
    images = [p for p in captured['input'][0]['content'] if p['type'] == 'input_image']
    assert len(images) == 2 and all(p['image_url'].startswith('data:image/png;base64,') for p in images)
    assert captured['model'] == 'gpt-4o' and captured['temperature'] == 0


def test_corrupt_annotation_is_reported_without_modifying_source(tmp_path):
    path = tmp_path / 'truncated.json'
    path.write_text('{"shapes": [')
    result = audit(tmp_path, tmp_path / 'audit.json')
    assert result['records'][0]['status'] == 'needs_review'
    assert path.read_text() == '{"shapes": ['


def test_baseline_comparison_uses_same_test_set_and_truth(tmp_path):
    manifest, _, target = records(tmp_path)
    result = experiment(manifest, tmp_path / 'runs', {"model": "synthetic"}, backend=SyntheticBackend())
    run = result['settings']['15']['runs'][0]
    write_json(tmp_path / 'baseline.json', {"targets": [{"record": target['record_path'],
               "predictions": str(resolve_path(tmp_path / 'runs/experiment.json', run['directory']) / 'predictions.json')}]})
    comparison = compare(tmp_path / 'runs/experiment.json', tmp_path / 'baseline.json', tmp_path / 'comparison')
    assert comparison['baseline']['legend_item']['f1'] == 1
    write_json(tmp_path / 'baseline.json', {"targets": []})
    with pytest.raises(ValueError, match="same target maps"):
        compare(tmp_path / 'runs/experiment.json', tmp_path / 'baseline.json', tmp_path / 'comparison')


@pytest.mark.parametrize('requested,returned', [
    ('gpt-4o-2024-11-20', 'gpt-4o-2024-11-20'),
    ('gpt-4o', 'gpt-4o-2024-08-06'),
    ('gpt-4o', None),
])
def test_comparison_labels_saved_model_identity(tmp_path, requested, returned):
    manifest, _, target = records(tmp_path)

    class IdentityBackend(SyntheticBackend):
        def infer(self, *args):
            predictions, raw = super().infer(*args)
            raw['model'] = returned
            return predictions, raw

    report = experiment(manifest, tmp_path / 'runs', {'model': requested},
                        counts=(15,), backend=IdentityBackend())
    run = report['settings']['15']['runs'][0]
    write_json(tmp_path / 'baseline.json', {
        'method': 'LayoutLMv3',
        'targets': [{'record': target['record_path'],
                     'predictions': str(resolve_path(tmp_path / 'runs/experiment.json', run['directory']) / 'predictions.json')}],
    })
    comparison = compare(tmp_path / 'runs/experiment.json', tmp_path / 'baseline.json', tmp_path / 'comparison')
    label = returned or requested
    assert comparison['experiment_method'] == label
    assert comparison['model_requested'] == requested
    assert comparison['models_returned'] == ([returned] if returned else [])
    assert comparison['in_context'] == comparison['gpt'] == report['settings']['15']['metrics']
    assert resolve_path(tmp_path / 'comparison/comparison.json', comparison['experiment']) == tmp_path / 'runs/experiment.json'
    assert resolve_path(tmp_path / 'comparison/comparison.json', comparison['baseline_manifest']) == tmp_path / 'baseline.json'
    table = (tmp_path / 'comparison/table.md').read_text()
    assert f'| {label} (15 ex.) |' in table
    assert '| LayoutLMv3 |' in table
    assert '| GPT (15 ex.) |' not in table


def test_source_frame_baseline_coordinates(tmp_path):
    _, _, target = records(tmp_path)
    prediction = {"target_map": target['map_id'], "coordinate_frame": "source_pixels",
                  "predictions": [{role: p[role] for role in ('legend_item', 'description')} for p in target['source_pairs']]}
    assert evaluate_saved(prediction, target)['linked_pair']['f1'] == 1


def test_incomplete_api_response_is_not_parsed_as_detections(tmp_path):
    _, example, target = records(tmp_path)
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: SimpleNamespace(
        output_text='{}', status='incomplete', model_dump=lambda **_: {"status": "incomplete"})))
    backend = OpenAIBackend({"model": "gpt-4o"}, client)
    with pytest.raises(ValueError, match="missing list"):
        infer(example, target, tmp_path / 'run', backend=backend)
    assert read_json(next((tmp_path / 'run.failed-attempts').glob('*/response.json')))['status'] == 'incomplete'


def test_real_sdk_serializes_responses_request_without_network(tmp_path):
    import httpx
    from openai import OpenAI
    _, example, target = records(tmp_path)
    def transport(request):
        body = json.loads(request.content)
        assert request.url.path == '/v1/responses'
        assert body['input'][0]['role'] == 'user'
        assert body['text']['format'] == {"type": "json_object"}
        return httpx.Response(200, json={"id": "resp_mock", "object": "response", "status": "completed", "model": "gpt-4o",
            "output": [{"id": "msg_mock", "type": "message", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": '{"predictions for synthetic_target_legend.png": []}', "annotations": []}]}]})
    with OpenAI(api_key='offline-test-key', http_client=httpx.Client(transport=httpx.MockTransport(transport))) as client:
        result, raw = OpenAIBackend({"model": "gpt-4o"}, client).infer(example, target, prompt(example, target, sample_pairs(example, 15, 0)))
    assert result == {"predictions for synthetic_target_legend.png": []}
    assert raw['id'] == 'resp_mock'


def test_lara_keeps_virtualenv_interpreter_symlink(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    module = repo / 'pipelines/segmentation/run_pipeline.py'
    module.parent.mkdir(parents=True)
    module.touch()
    interpreter = tmp_path / 'environment/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to('/usr/bin/python3')
    (tmp_path / 'weights').mkdir()
    (tmp_path / 'images').mkdir()
    for name in ['config.yaml', 'config.json', 'model_final.pth']:
        (tmp_path / 'weights' / name).touch()
    captured = {}
    def run(command, **kwargs):
        captured.update(command=command, **kwargs)
    monkeypatch.setattr('legend_icl.prepare.subprocess.run', run)
    run_lara(repo, interpreter, tmp_path / 'weights', tmp_path / 'images', tmp_path / 'out', tmp_path / 'work', cpu=True)
    assert captured['command'][0] == str(interpreter)
    assert '--no_gpu' in captured['command']
    assert captured['cwd'] == repo


def test_nonfinite_response_rejections_can_be_saved(tmp_path):
    _, example, target = records(tmp_path)
    class BadBoxBackend:
        def infer(self, *args):
            return {f"predictions for {target['map_id']}_legend.png": [
                {'legend_item': [0, 0, float('nan'), 1], 'description': [0, 0, 1, 1]}]}, {'id': 'offline'}
    result = infer(example, target, tmp_path / 'run', backend=BadBoxBackend())
    assert result['predictions'] == []
    assert len(result['rejected']) == 1
    assert read_json(tmp_path / 'run/predictions.json')['status'] == 'completed'
