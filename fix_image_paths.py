#!/usr/bin/env python
"""
Fix image paths for Do ∪ Dl (Phase 2 pretraining).

The concatenated annotation file contains both optical and SAR images,
but they live in different directories. This script either:
  1. Creates symlinks to unify image locations
  2. Modifies annotation file_name fields to use absolute paths

Usage:
    python fix_image_paths.py \
        --optical-img /path/to/ShipRSImageNet/images \
        --sar-img /path/to/SSDD/JPEGImages \
        --ann-dir /path/to/data/optical_ssdd/annotations \
        --mode symlink
"""

import argparse
import json
import os
import glob


def fix_annotations_abs_path(ann_path, optical_img_dir, sar_img_dir, sar_prefix="sar_"):
    """Modify file_name in annotations to use absolute paths."""
    with open(ann_path, "r") as f:
        anno = json.load(f)

    optical_files = set(os.listdir(optical_img_dir))
    
    modified = 0
    for img in anno["images"]:
        fname = img["file_name"]
        # If the file exists in optical dir, use optical path
        if fname in optical_files or os.path.exists(os.path.join(optical_img_dir, fname)):
            img["file_name"] = os.path.join(optical_img_dir, fname)
        else:
            # Otherwise assume it's a SAR image
            # Try with and without prefix
            sar_name = fname.replace(sar_prefix, "") if fname.startswith(sar_prefix) else fname
            if os.path.exists(os.path.join(sar_img_dir, sar_name)):
                img["file_name"] = os.path.join(sar_img_dir, sar_name)
            elif os.path.exists(os.path.join(sar_img_dir, fname)):
                img["file_name"] = os.path.join(sar_img_dir, fname)
            else:
                print(f"  WARNING: Cannot find image {fname} in either directory")
        modified += 1

    with open(ann_path, "w") as f:
        json.dump(anno, f, ensure_ascii=False)

    print(f"  Fixed {modified} image paths in {ann_path}")


def create_symlinks(optical_img_dir, sar_img_dir, output_dir):
    """Create a unified image directory with symlinks."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Link all optical images
    count = 0
    for f in os.listdir(optical_img_dir):
        src = os.path.join(optical_img_dir, f)
        dst = os.path.join(output_dir, f)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
            count += 1
    
    # Link all SAR images with prefix
    for f in os.listdir(sar_img_dir):
        src = os.path.join(sar_img_dir, f)
        dst = os.path.join(output_dir, f"sar_{f}")
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
            count += 1
    
    print(f"  Created {count} symlinks in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optical-img", type=str, required=True)
    parser.add_argument("--sar-img", type=str, required=True)
    parser.add_argument("--ann-dir", type=str, required=True,
                        help="Directory containing Do ∪ Dl annotation files")
    parser.add_argument("--mode", type=str, choices=["symlink", "abspath"],
                        default="abspath",
                        help="symlink: create unified dir; abspath: modify file_name")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir for symlink mode")
    args = parser.parse_args()

    if args.mode == "symlink":
        out = args.output_dir or os.path.join(os.path.dirname(args.ann_dir), "images")
        create_symlinks(args.optical_img, args.sar_img, out)
    else:
        ann_files = glob.glob(os.path.join(args.ann_dir, "*.json"))
        for ann_file in sorted(ann_files):
            print(f"Processing: {ann_file}")
            fix_annotations_abs_path(ann_file, args.optical_img, args.sar_img)


if __name__ == "__main__":
    main()
