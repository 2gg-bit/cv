"""Verify real Phase 1/2 checkpoints on CPU, without datasets or training."""

import argparse

from mmcv import Config, DictAction
from mmdet.models import build_detector

from ssod.utils import get_root_logger, patch_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg = patch_config(cfg)
    if cfg.model.type != "DualTeacher":
        raise ValueError("This check requires a DualTeacher config")
    if cfg.get("load_from") or cfg.get("resume_from"):
        raise ValueError("This is a fresh-run check; set load_from/resume_from=None")
    logger = get_root_logger(log_level=cfg.get("log_level", "INFO"))
    model = build_detector(cfg.model)
    model.init_weights()
    model.init_from_pretrained()
    logger.info("CPU initialization check passed; no dataset, optimizer or training was run.")


if __name__ == "__main__":
    main()
