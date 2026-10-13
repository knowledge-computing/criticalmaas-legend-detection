import argparse
import json
import sys
from pathlib import Path

from .annotations import audit
from .dataset import extract, validate_dataset
from .comparison import compare, evaluate_saved
from .experiments import experiment, recompute
from .inference import infer, load_record, overlay
from .io import read_json, write_json
from .prepare import prepare, run_lara


def load_config(path):
    if not path:
        return {"seed": 1337, "icl": {"num_examples": 15, "normalize_coords": False},
                "openai": {"model": "gpt-4o", "temperature": 0., "api": "responses", "output_format": "json_object"}}
    import yaml
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML object")
    return config


def parser():
    root = argparse.ArgumentParser(description="GeoSearch 2025 legend item-description pipeline")
    commands = root.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser('extract', help='Recover original scans embedded in LabelMe annotation JSON')
    cmd.add_argument('--directory', required=True)
    cmd.add_argument('--output', required=True)
    cmd = commands.add_parser('validate-dataset', help='Check concrete image and annotation hashes')
    cmd.add_argument('--manifest', required=True)
    cmd.add_argument('--data-root')
    cmd = commands.add_parser("audit", help="Inspect annotation formats and pairing problems")
    cmd.add_argument("directory")
    cmd.add_argument("--output", required=True)
    cmd = commands.add_parser("segment", help="Run the actual external LARA segmentation pipeline")
    for name in ["repo", "python", "model", "images", "output", "workdir"]:
        cmd.add_argument("--" + name, required=True)
    cmd.add_argument("--cpu", action="store_true")
    cmd.add_argument("--revision")
    cmd = commands.add_parser("prepare", help="Crop a map and translate annotations")
    cmd.add_argument("--image", required=True)
    cmd.add_argument("--annotation")
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--map-id")
    cmd.add_argument("--replace", action="store_true")
    group = cmd.add_mutually_exclusive_group(required=True)
    group.add_argument("--crop", nargs=4, type=float)
    group.add_argument("--segmentation")
    group.add_argument("--already-cropped", action="store_true")
    cmd.add_argument("--merge-groups", action="store_true")
    cmd.add_argument("--min-confidence", type=float, default=0.25)
    for name in ["infer", "experiment"]:
        cmd = commands.add_parser(name)
        if name == "infer":
            cmd.add_argument("--example", required=True)
            cmd.add_argument("--target", required=True)
            cmd.add_argument("--num-examples", type=int)
        else:
            cmd.add_argument("--manifest", required=True)
            cmd.add_argument("--counts", nargs="+", type=int, default=[5, 10, 15, 20])
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--config")
        cmd.add_argument("--model")
        cmd.add_argument("--seed", type=int)
        cmd.add_argument("--normalized", action="store_true", default=None)
        cmd.add_argument("--dry-run", action="store_true")
        cmd.add_argument("--resume", action="store_true")
        cmd.add_argument("--replace", action="store_true")
        if name == 'experiment':
            cmd.add_argument('--data-root')
            cmd.add_argument('--protocol', choices=['manuscript'])
            cmd.add_argument('--run-cache', help='Save and verify each completed map/count across interrupted attempts')
    cmd = commands.add_parser("evaluate", help="Evaluate saved GPT or external baseline predictions")
    cmd.add_argument("--predictions", required=True)
    cmd.add_argument("--target", required=True)
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--threshold", type=float, default=0.5)
    cmd = commands.add_parser("compare", help="Compare external baseline predictions on the same test maps")
    cmd.add_argument("--experiment", required=True)
    cmd.add_argument("--baseline-manifest", required=True)
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--count", type=int, default=15)
    cmd = commands.add_parser("visualize")
    cmd.add_argument("--predictions", required=True)
    cmd.add_argument("--target", required=True)
    cmd.add_argument("--output", required=True)
    cmd = commands.add_parser('recompute', help='Recompute tables from a portable verified experiment bundle')
    cmd.add_argument('--experiment', required=True)
    cmd.add_argument('--output', required=True)
    cmd = commands.add_parser("smoke", help="Offline synthetic integration run; not paper results")
    cmd.add_argument("--output", required=True)
    return root


def dispatch(args):
    values = vars(args).copy()
    command = values.pop("command")
    if command == 'extract':
        result = extract(args.directory, args.output)
        return {'maps': len(result['maps']), 'included': sum(r['status'] == 'included' for r in result['maps'])}
    if command == 'validate-dataset':
        return {'status': 'verified', 'maps': len(validate_dataset(args.manifest, args.data_root)['maps'])}
    if command == "audit":
        return audit(args.directory, args.output)
    if command == "segment":
        run_lara(**values)
        return {"status": "LARA command completed; inspect exported segmentation files"}
    if command == "prepare":
        return {"record": str(prepare(**values))}
    if command == 'recompute':
        return recompute(args.experiment, args.output)
    if command == "compare":
        return compare(args.experiment, args.baseline_manifest, args.output, args.count)
    if command in {"infer", "experiment"}:
        cfg = load_config(args.config)
        backend_cfg = cfg.get("inference", cfg.get("openai", {}))
        if args.model:
            backend_cfg = {**backend_cfg, "model": args.model}
        seed = args.seed if args.seed is not None else cfg.get("seed", 1337)
        normalized = args.normalized if args.normalized is not None else cfg.get("icl", {}).get("normalize_coords", False)
        if command == "infer":
            number = args.num_examples if args.num_examples is not None else cfg.get("icl", {}).get("num_examples", 15)
            result = infer(load_record(args.example), load_record(args.target), args.output,
                           number, seed, backend_cfg, normalized, args.dry_run, resume=args.resume, replace=args.replace)
            return {"status": result["status"], "output": args.output, "num_examples": number}
        result = experiment(args.manifest, args.output, backend_cfg, args.counts, seed,
                            normalized, args.dry_run, args.resume, replace=args.replace, data_root=args.data_root, protocol=args.protocol,
                            run_cache=args.run_cache)
        return {"status": result["status"], "output": args.output, "counts": args.counts, "maps": len(result["target_maps"])}
    if command in {"evaluate", "visualize"}:
        prediction = read_json(args.predictions)
        target = load_record(args.target)
        if prediction.get("target_map") != target["map_id"]:
            raise ValueError("Prediction target_map must match the target record")
        if command == "visualize":
            if prediction.get("coordinate_frame") != "crop_pixels":
                raise ValueError("Overlay predictions must use crop_pixels")
            overlay(target["image"], prediction["predictions"], args.output)
            return {"overlay": args.output}
        metrics = evaluate_saved(prediction, target, args.threshold)
        write_json(args.output, metrics)
        return metrics
    if command == "smoke":
        from .smoke import smoke
        return smoke(args.output)
    raise ValueError(command)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
