"""Two-image ICL requests with a saved prompt, strict local validation and provenance."""
import base64
import hashlib
import io
import json
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw

from .artifacts import attempt, snapshot_record, verify
from .provenance import canonical_hash, environment, file_hash, semantic_record, source_identity, truth_identity, now
from .geometry import box, offset
from .io import read_json, resolve_path, write_json


def load_record(path, data_root=None, require_image=True):
    path = Path(path).resolve()
    record = read_json(path)
    if record.get("schema_version") != 1 or record.get("coordinate_frame") != "crop_pixels":
        raise ValueError(f"Not a prepared legend record: {path}")
    image_value = record.get("image")
    image_path = resolve_path(path, image_value) if image_value else None
    if image_path and data_root and not image_path.exists():
        image_path = Path(data_root) / image_value
    record = {**record, "record_path": str(path), "image": str(image_path) if image_path else None}
    if require_image and image_path:
        with Image.open(image_path) as image:
            if list(image.size) != record["image_size"]:
                raise ValueError(f"Prepared image dimensions changed: {image_path}")
    elif require_image and record.get("preparation_status") != "no_legend":
        raise ValueError("Prepared record has no image and is not a valid zero-region outcome")
    for pair in record["pairs"]:
        for role in ("legend_item", "description"):
            box(pair[role], record["image_size"])
    return record


def same_map(example, target):
    return (example["map_id"] == target["map_id"] or
            bool(example.get("source_image_sha256") and example.get("source_image_sha256") == target.get("source_image_sha256")) or
            Path(example["source_image"]).resolve() == Path(target["source_image"]).resolve())


def sample_pairs(example, number, seed):
    if number <= 0 or len(example["pairs"]) < number:
        raise ValueError(f"Example {example['map_id']} has {len(example['pairs'])} visible pairs; exactly {number} are required")
    return random.Random(seed).sample(example["pairs"], number)


def prompt(example, target, pairs, normalized=False):
    rendered = []
    for p in pairs:
        result = {}
        for role in ("legend_item", "description"):
            b = p[role]
            result[role] = [v / example["image_size"][i % 2] for i, v in enumerate(b)] if normalized else b
        rendered.append(result)
    return {"task": "Given a scanned map legend area, detect legend items and their descriptions coordinates",
            f"examples from {example['map_id']}_legend.png": rendered,
            f"predictions for {target['map_id']}_legend.png": [
                {"legend_item": ["??"] * 4, "description": ["??"] * 4}]}


def image_url(path):
    # Prepared legends are already PNG: preserve those exact encoded bytes.
    path = Path(path)
    if path.suffix.lower() == '.png':
        payload = path.read_bytes()
        if payload.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'data:image/png;base64,' + base64.b64encode(payload).decode('ascii')
    # Convert other source formats without changing their dimensions.
    with Image.open(path) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def response_schema(key):
    coordinate = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
    return {"type": "object", "properties": {key: {"type": "array", "items": {
        "type": "object", "properties": {"legend_item": coordinate, "description": coordinate},
        "required": ["legend_item", "description"], "additionalProperties": False}}},
        "required": [key], "additionalProperties": False}


def request_parts(example, target, value, normalized=False, style="manuscript"):
    if style == 'manuscript':
        if normalized:
            raise ValueError('The manuscript prompt uses pixel coordinates; normalized experiments need prompt_style explicit')
        return [{'type': 'input_image', 'image_url': image_url(example['image']), 'detail': 'high'},
                {'type': 'input_text', 'text': json.dumps(value) + '\nReturn the predictions object as JSON.'},
                {'type': 'input_image', 'image_url': image_url(target['image']), 'detail': 'high'}]
    if style != 'explicit':
        raise ValueError('prompt_style must be manuscript or explicit')
    coords = "normalized to [0, 1] separately by image width and height" if normalized else "in pixels of each displayed legend crop"
    return [
        {"type": "input_text", "text": f"Example legend: {example['map_id']}_legend.png; dimensions {example['image_size']}."},
        {"type": "input_image", "image_url": image_url(example["image"]), "detail": "high"},
        {"type": "input_text", "text": json.dumps(value) + f"\nCoordinates are {coords}, in [xmin, ymin, xmax, ymax] order with origin at the top-left. Return only the predictions key as JSON; detect all target pairs, not only the number shown as examples."},
        {"type": "input_text", "text": f"Target legend: {target['map_id']}_legend.png; dimensions {target['image_size']}."},
        {"type": "input_image", "image_url": image_url(target["image"]), "detail": "high"}]


class OpenAIBackend:
    def __init__(self, config, client=None):
        self.config = config
        if client is None:
            dotenv = Path(__file__).resolve().parents[2] / '.env'
            if not os.getenv('OPENAI_API_KEY') and dotenv.exists():
                for line in dotenv.read_text().splitlines():
                    key, separator, value = line.partition('=')
                    if separator and key.strip() == 'OPENAI_API_KEY':
                        os.environ['OPENAI_API_KEY'] = value.strip().strip('\"').strip("'")
                        break
        if client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("Set OPENAI_API_KEY to run GPT inference. Use --dry-run for offline prompt preparation.")
            from openai import OpenAI
            client = OpenAI(timeout=config.get("timeout", 180), max_retries=config.get("max_retries", 2))
        self.client = client

    def infer(self, example, target, value, normalized=False):
        parts = request_parts(example, target, value, normalized, self.config.get("prompt_style", "manuscript"))
        key = f"predictions for {target['map_id']}_legend.png"
        fmt = {"type": "json_object"}
        if self.config.get("output_format", "json_object") == "json_schema":
            fmt = {"type": "json_schema", "name": "legend_pairs", "strict": True, "schema": response_schema(key)}
        elif self.config.get("output_format", "json_object") != "json_object":
            raise ValueError("output_format must be json_object or json_schema")
        kwargs = {"model": self.config.get("model", "gpt-4o"), "temperature": self.config.get("temperature", 0.)}
        api = self.config.get("api", "responses")
        if api == "responses":
            response = self.client.responses.create(**kwargs, input=[{"role": "user", "content": parts}],
                                                    text={"format": fmt}, store=False,
                                                    max_output_tokens=self.config.get("max_output_tokens", 16000))
            raw = response.model_dump(mode="json")
            text = response.output_text
            if response.status != "completed":
                return {"error": f"Response status {response.status}"}, raw
        elif api == "chat_completions":
            content = []
            for part in parts:
                if part["type"] == "input_text":
                    content.append({"type": "text", "text": part["text"]})
                else:
                    content.append({"type": "image_url", "image_url": {"url": part["image_url"], "detail": part["detail"]}})
            chat_format = fmt if fmt["type"] == "json_object" else {"type": "json_schema", "json_schema": {k: v for k, v in fmt.items() if k != "type"}}
            response = self.client.chat.completions.create(**kwargs, messages=[{"role": "user", "content": content}],
                response_format=chat_format, max_completion_tokens=self.config.get("max_output_tokens", 16000), store=False)
            raw = response.model_dump(mode="json")
            choice = response.choices[0]
            if choice.finish_reason != "stop" or choice.message.refusal:
                return {"error": f"Response did not complete: {choice.finish_reason}; refusal={choice.message.refusal}"}, raw
            text = choice.message.content
        else:
            raise ValueError("api must be responses or chat_completions")
        try:
            return json.loads(text), raw
        except (ValueError, TypeError):
            return {"error": "Response was not valid JSON", "text": text}, raw


def decode_predictions(value, target, normalized=False):
    key = f"predictions for {target['map_id']}_legend.png"
    if not isinstance(value, dict) or key not in value or not isinstance(value[key], list):
        raise ValueError(f"Model output is missing list {key!r}")
    accepted, rejected = [], []
    for index, pair in enumerate(value[key]):
        try:
            if not isinstance(pair, dict):
                raise ValueError("Pair is not an object")
            result = {}
            for role in ("legend_item", "description"):
                b = box(pair[role], [1, 1] if normalized else target["image_size"])
                result[role] = [v * target["image_size"][i % 2] for i, v in enumerate(b)] if normalized else b
            accepted.append(result)
        except (ValueError, KeyError, TypeError) as exc:
            # The full raw response is saved separately; nonfinite values must not
            # make the rejection log itself impossible to serialize as valid JSON.
            rejected.append({"index": index, "error": str(exc)})
    return accepted, rejected


def effective_request(example, target, value, config, normalized):
    if target.get('preparation_status') == 'no_legend':
        return {'kind': 'valid_zero_region', 'request': None}
    if config.get('backend', 'openai') != 'openai':
        raise ValueError('backend must be openai')
    parts = request_parts(example, target, value, normalized, config.get("prompt_style", "manuscript"))
    for part in parts:
        if part['type'] == 'input_image':
            part['image_url'] = 'sha256:' + hashlib.sha256(part['image_url'].encode()).hexdigest()
    key = f"predictions for {target['map_id']}_legend.png"
    return {'parts': parts, 'model': config.get('model', 'gpt-4o'),
            'temperature': config.get('temperature', 0.), 'api': config.get('api', 'responses'),
            'output_format': config.get('output_format', 'json_object'),
            'schema': response_schema(key) if config.get('output_format') == 'json_schema' else None,
            'max_output_tokens': config.get('max_output_tokens', 16000), 'store': False}


def fingerprint(example, target, value, config, normalized):
    inputs = {'example': semantic_record(example), 'target': semantic_record(target),
              'example_image': file_hash(example['image']),
              'target_image': file_hash(target['image']) if target.get('image') else None,
              'request': effective_request(example, target, value, config, normalized),
              'code': source_identity()['source_digest'], 'dependencies': environment()['packages'],
              'schema_versions': {'request': 2, 'decoding': 2, 'preparation': 2}, 'normalized': normalized}
    return canonical_hash(inputs)


def make_backend(config):
    kind = config.get('backend', 'openai')
    if kind != 'openai':
        raise ValueError('backend must be openai')
    return OpenAIBackend(config)


class DeferredBackend:
    """Reuse one loaded model/client, initializing only when inference is needed."""
    def __init__(self, config):
        self.config = config
        self.instance = None

    def infer(self, *args):
        if self.instance is None:
            self.instance = make_backend(self.config)
        return self.instance.infer(*args)


def overlay(image_path, pairs, output):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(1, round(min(image.size) / 500))
    for index, pair in enumerate(pairs, 1):
        for role, color in [("legend_item", "red"), ("description", "blue")]:
            draw.rectangle(tuple(pair[role]), outline=color, width=width)
            draw.text(tuple(pair[role][:2]), str(index), fill=color)
        item, description = pair["legend_item"], pair["description"]
        draw.line(((item[0]+item[2])/2, (item[1]+item[3])/2,
                   (description[0]+description[2])/2, (description[1]+description[3])/2), fill="green", width=width)
    image.thumbnail((2400, 2400))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    image.save(output)



def infer(example, target, output, number=15, seed=1337, config=None,
          normalized=False, dry_run=False, backend=None, selected_pairs=None, resume=False, replace=False):
    if same_map(example, target):
        raise ValueError("Example and target must be different maps")
    config = config or {"model": "gpt-4o", "temperature": 0., "api": "responses", "output_format": "json_object"}
    pairs = selected_pairs if selected_pairs is not None else sample_pairs(example, number, seed)
    if len(pairs) != number:
        raise ValueError("Selected example count does not match the requested setting")
    value = prompt(example, target, pairs, normalized)
    signature = fingerprint(example, target, value, config, normalized)
    output = Path(output)
    if resume and (output / "predictions.json").is_file():
        verify(output)
        cached = read_json(output / "predictions.json")
        if cached.get("fingerprint") != signature:
            raise ValueError(f"Cached result has different inputs/configuration: {output}; use a new output directory")
        return cached
    with attempt(output, replace=replace) as staging:
        example_copy = load_record(snapshot_record(example, staging / 'inputs', 'example'))
        target_copy = load_record(snapshot_record(target, staging / 'inputs', 'target'))
        write_json(staging / 'prompt.json', value)
        truth = truth_identity(target)
        write_json(staging / 'evaluation_truth.json', truth)
        run = {"schema_version": 2, "fingerprint": signature, "example_map": example["map_id"], "target_map": target["map_id"],
               "example_record": "inputs/example.json", "target_record": "inputs/target.json",
               "evaluation_truth": "evaluation_truth.json", "evaluation_truth_sha256": canonical_hash(truth),
               "model_requested": config.get("model", "gpt-4o"), "api": config.get("api", "responses"),
               "configuration": config,
               "num_examples": number, "seed": seed, "normalized": normalized,
               "example_pair_ids": [p.get("pair_id") for p in pairs], "crop_bbox": target["crop_bbox"],
               "request_image_order": [example["map_id"], target["map_id"]], "created_at": now(),
               "source": source_identity(), "environment": environment(),
               "effective_request": effective_request(example, target, value, config, normalized)}
        run['inference_kind'] = ('valid_zero_region' if target.get('preparation_status') == 'no_legend'
                                 else 'model_request')
        if dry_run:
            run['status'] = 'dry_run'
            write_json(staging / 'run.json', run)
            return run
        try:
            if target.get('preparation_status') == 'no_legend':
                raw, predictions, rejected = {'model': None, 'id': None}, [], []
                run['zero_region'] = True
            else:
                model_value, raw = (backend or make_backend(config)).infer(example_copy, target_copy, value, normalized)
                write_json(staging / 'response.json', raw)
                predictions, rejected = decode_predictions(model_value, target_copy, normalized)
            dx, dy = target["crop_bbox"][:2]
            full = [{role: offset(pair[role], dx, dy) for role in ("legend_item", "description")} for pair in predictions]
            run.update(status="completed", model_returned=raw.get("model"), response_id=raw.get("id"),
                       usage=raw.get("usage"), predictions=predictions, source_predictions=full, rejected=rejected,
                       coordinate_frame="crop_pixels")
            from .metrics import evaluate
            if target['has_ground_truth']:
                write_json(staging / 'metrics.json', evaluate(full, truth['source_pairs'], invalid_predictions=len(rejected)))
            write_json(staging / 'predictions.json', run)
            if target.get('image'):
                overlay(target_copy["image"], predictions, staging / "overlay.png")
            return run
        except Exception as exc:
            write_json(staging / "run.json", {**run, "status": "failed", "error": str(exc)})
            raise
