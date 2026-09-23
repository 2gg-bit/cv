"""Freeze a user-selected, valid B0 and generate four isolated experiments."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_config(values, path):
    from mmcv import Config
    # MMCV 1.3.9 Config.dump dereferences self.filename, which is None for
    # these generated configs. Keep its Python formatter, without that branch.
    text = Config(values).pretty_text
    with open(str(path), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def annotation_paths(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "ann_file":
                for path in value if isinstance(value, (list, tuple)) else [value]:
                    yield Path(path).resolve()
            else:
                for path in annotation_paths(value):
                    yield path
    elif isinstance(data, (list, tuple)):
        for value in data:
            for path in annotation_paths(value):
                yield path


def main():
    from mmcv import Config, DictAction
    from ssod.utils import patch_config
    from ssod.utils.ablation import make_suite, differences
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Correct B0 config or resolved config, never selected automatically")
    parser.add_argument("--out-dir", required=True, help="New config/manifest directory")
    parser.add_argument("--work-root", required=True, help="New directory for training outputs")
    parser.add_argument("--seed", type=int, required=True, help="Same seed as B0")
    parser.add_argument("--fg-weight", type=float, default=0.1)
    parser.add_argument("--pg-start-ratio", type=float, default=0.5)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from this checkout root so relative B0 data/weight paths keep their meaning")
    output, work = Path(args.out_dir).resolve(), Path(args.work_root).resolve()
    if output.exists() or work.exists():
        raise ValueError("Config and work directories must be new; nothing is overwritten")
    cfg = Config.fromfile(args.baseline)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg = patch_config(cfg)
    baseline = cfg._cfg_dict.to_dict()
    suite = make_suite(baseline, work, args.seed, args.fg_weight, args.pg_start_ratio)
    weights = {}
    for key in ("load1_from", "load2_from"):
        path = Path(baseline["model"]["train_cfg"][key]).resolve()
        weights[key] = dict(path=str(path), sha256=file_hash(path))
    annotations = {str(p): file_hash(p) for p in annotation_paths(baseline["data"])}
    output.mkdir(parents=True)
    dump_config(baseline, output / "baseline_resolved.py")
    manifest = dict(baseline=str(Path(args.baseline).resolve()),
                    baseline_sha256=file_hash(args.baseline), seed=args.seed,
                    cfg_options=args.cfg_options, initialization=weights, annotations=annotations,
                    baseline_resolved_sha256=file_hash(output / "baseline_resolved.py"),
                    experiments={})
    for name, item in suite.items():
        target = output / (name + ".py")
        dump_config(item, target)
        manifest["experiments"][name] = dict(
            config=str(target), sha256=file_hash(target),
            changes_vs_b0=differences(baseline, item))
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Prepared B0 snapshot and four experiments: " + str(output))
    print("Review manifest.json. No training was started; existing B0 does not need rerunning.")


if __name__ == "__main__":
    main()
