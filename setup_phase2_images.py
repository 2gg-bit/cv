#!/usr/bin/env python
"""
Create unified image directory for Phase 2 (Do ∪ Dl) pretraining.

Merges optical and SAR images into a single directory so that
CocoDataset can load them with a single img_prefix.

Usage:
    python setup_phase2_images.py \
        --optical-img /path/to/ShipRSImageNet/images \
        --sar-img /path/to/SSDD/JPEGImages \
        --ann-dir /path/to/data/optical_ssdd/annotations \
        --output-dir /path/to/data/optical_ssdd/images
"""

import argparse
import json
import os
import glob
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optical-img", type=str, required=True,
                        help="Optical image directory")
    parser.add_argument("--sar-img", type=str, required=True,
                        help="SAR image directory")
    parser.add_argument("--ann-dir", type=str, required=True,
                        help="Do ∪ Dl annotation directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output unified image directory")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink",
                        help="symlink (saves disk space) or copy")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get all existing filenames in output dir
    existing = set(os.listdir(args.output_dir))

    # ---- Step 1: Link/copy optical images (original names) ----
    count_opt = 0
    opt_files = set()
    for fname in sorted(os.listdir(args.optical_img)):
        opt_files.add(fname)
        if fname not in existing:
            src = os.path.abspath(os.path.join(args.optical_img, fname))
            dst = os.path.join(args.output_dir, fname)
            if args.mode == "symlink":
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)
            count_opt += 1

    # ---- Step 2: Link/copy SAR images ----
    # SAR images might have filename collisions with optical images.
    # If collision, prefix with "sar_".
    count_sar = 0
    sar_name_map = {}  # original_name -> name_in_output
    for fname in sorted(os.listdir(args.sar_img)):
        if fname in opt_files or fname in existing:
            # Collision: rename
            base, ext = os.path.splitext(fname)
            new_name = f"sar_{fname}"
        else:
            new_name = fname

        sar_name_map[fname] = new_name
        if new_name not in existing:
            src = os.path.abspath(os.path.join(args.sar_img, fname))
            dst = os.path.join(args.output_dir, new_name)
            if args.mode == "symlink":
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)
            count_sar += 1

    print(f"Optical images: {count_opt} added")
    print(f"SAR images: {count_sar} added ({len(sar_name_map)} total)")

    # ---- Step 3: Update annotation file_name fields ----
    ann_files = glob.glob(os.path.join(args.ann_dir, "*.json"))
    for ann_path in sorted(ann_files):
        with open(ann_path, "r") as f:
            anno = json.load(f)

        modified = 0
        for img in anno.get("images", []):
            fname = img["file_name"]
            # If it's a SAR image that was renamed
            if fname in sar_name_map:
                img["file_name"] = sar_name_map[fname]
                modified += 1
            elif os.path.basename(fname) in sar_name_map:
                # Handle path prefixes
                img["file_name"] = sar_name_map[os.path.basename(fname)]
                modified += 1

        if modified > 0:
            with open(ann_path, "w") as f:
                json.dump(anno, f, ensure_ascii=False)
            print(f"Updated {modified} SAR paths in {os.path.basename(ann_path)}")

    print(f"\nDone! Unified image directory: {args.output_dir}")
    print(f"Total images: {len(os.listdir(args.output_dir))}")


if __name__ == "__main__":
    main()
