#!/usr/bin/env python
"""
Prepare ShipRSImageNet + SSDD data for Dual Teacher reproduction.

This script generates:
1. Few-shot SSDD splits: labeled (1/3/5/10) + unlabeled
2. ShipRSImageNet annotation (sup1: optical only)
3. Do ∪ Dl concatenation: optical + few-shot SAR (for pretrain phase 2)

Usage:
    python prepare_data.py \
        --optical-ann /path/to/ShipRSImageNet/annotations.json \
        --optical-img /path/to/ShipRSImageNet/images/ \
        --sar-ann /path/to/SSDD/annotations/train.json \
        --sar-img /path/to/SSDD/JPEGImages/ \
        --sar-test-ann /path/to/SSDD/annotations/test.json \
        --output-dir ./data \
        --shots 1 3 5 10
"""

import argparse
import json
import os
import shutil
import numpy as np
from copy import deepcopy


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {path} ({len(data.get('images', []))} images, "
          f"{len(data.get('annotations', []))} annotations)")


def normalize_categories(anno, target_cat_name="ship"):
    """Ensure all categories are normalized to a single 'ship' class."""
    if len(anno["categories"]) == 1 and anno["categories"][0]["name"].lower() in (
        "ship", "ships", "boat", "boats"
    ):
        return anno  # already single-class ship

    # Merge all categories into one "ship" class
    new_cat = [{"id": 1, "name": target_cat_name, "supercategory": "vehicle"}]
    cat_mapping = {cat["id"]: 1 for cat in anno["categories"]}

    new_anno = deepcopy(anno)
    new_anno["categories"] = new_cat
    for ann in new_anno["annotations"]:
        ann["category_id"] = cat_mapping.get(ann["category_id"], 1)
    return new_anno


def prepare_optical_annotations(optical_ann_path, output_dir):
    """
    Step 1: Prepare ShipRSImageNet annotations as sup1 (optical domain Do).
    """
    print("\n[Step 1] Preparing optical (ShipRSImageNet) annotations...")
    anno = load_json(optical_ann_path)
    anno = normalize_categories(anno)

    # Save as Do (optical only)
    out_path = os.path.join(output_dir, "optical", "dior_annotations.json")
    save_json(anno, out_path)
    return anno


def prepare_ssdd_fewshot(sar_ann_path, output_dir, shots, num_folds=5):
    """
    Step 2: Generate few-shot SSDD splits.
    For each shot count and fold:
      - labeled: `shots` randomly selected SAR images with annotations
      - unlabeled: remaining SAR images (no annotations)
    """
    print(f"\n[Step 2] Preparing SSDD few-shot splits (shots={shots})...")
    anno = load_json(sar_ann_path)
    anno = normalize_categories(anno)

    images = anno["images"]
    annotations = anno["annotations"]

    # Build image_id -> annotations mapping
    img_id_to_anns = {}
    for ann in annotations:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    for shot in shots:
        for fold in range(1, num_folds + 1):
            seed = fold * 1000 + shot
            np.random.seed(seed)

            n_images = len(images)
            labeled_indices = np.random.choice(n_images, size=shot, replace=False)
            labeled_indices_set = set(labeled_indices)

            labeled_images = []
            unlabeled_images = []
            labeled_ids = set()

            for i, img in enumerate(images):
                if i in labeled_indices_set:
                    labeled_images.append(img)
                    labeled_ids.add(img["id"])
                else:
                    unlabeled_images.append(img)

            # Collect annotations for labeled images only
            labeled_annotations = []
            for ann in annotations:
                if ann["image_id"] in labeled_ids:
                    labeled_annotations.append(ann)

            semi_dir = os.path.join(output_dir, "ssdd", "annotations", "semi_supervised")
            os.makedirs(semi_dir, exist_ok=True)

            # Save labeled split: Dl
            labeled_name = f"instances_train2017.{fold}@{shot}.json"
            save_json(
                {"images": labeled_images, "annotations": labeled_annotations,
                 "categories": anno["categories"]},
                os.path.join(semi_dir, labeled_name),
            )

            # Save unlabeled split: Du
            unlabeled_name = f"instances_train2017.{fold}@{shot}-unlabeled.json"
            save_json(
                {"images": unlabeled_images, "annotations": [],
                 "categories": anno["categories"]},
                os.path.join(semi_dir, unlabeled_name),
            )

            print(f"  Fold {fold}, {shot}-shot: "
                  f"{len(labeled_images)} labeled / {len(unlabeled_images)} unlabeled")

    # Also save full training annotation (for reference)
    save_json(anno, os.path.join(output_dir, "ssdd", "annotations", "train.json"))


def prepare_sar_test(sar_test_ann_path, output_dir):
    """Step 3: Copy SAR test annotations."""
    print("\n[Step 3] Preparing SSDD test annotations...")
    if sar_test_ann_path and os.path.exists(sar_test_ann_path):
        anno = load_json(sar_test_ann_path)
        anno = normalize_categories(anno)
        save_json(anno, os.path.join(output_dir, "ssdd", "annotations", "test.json"))
    else:
        print("  WARNING: No SAR test annotation provided. You need to prepare it manually.")


def prepare_do_union_dl(optical_anno, sar_ann_path, output_dir, shots, num_folds=5):
    """
    Step 4: Generate Do ∪ Dl (optical + few-shot SAR).
    This is used for pretraining T2/S2 in Phase 2.
    """
    print(f"\n[Step 4] Preparing Do ∪ Dl (optical + few-shot SAR)...")
    sar_anno = load_json(sar_ann_path)
    sar_anno = normalize_categories(sar_anno)

    opt_images = optical_anno["images"]
    opt_annotations = optical_anno["annotations"]
    sar_images = sar_anno["images"]
    sar_annotations = sar_anno["annotations"]

    # Build SAR image_id -> annotations
    sar_img_id_to_anns = {}
    for ann in sar_annotations:
        sar_img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    for shot in shots:
        for fold in range(1, num_folds + 1):
            seed = fold * 1000 + shot
            np.random.seed(seed)

            n_sar_images = len(sar_images)
            labeled_indices = np.random.choice(n_sar_images, size=shot, replace=False)
            labeled_indices_set = set(labeled_indices)

            # Get selected SAR images and their annotations
            selected_sar_images = []
            selected_sar_ids = set()
            for i, img in enumerate(sar_images):
                if i in labeled_indices_set:
                    selected_sar_images.append(img)
                    selected_sar_ids.add(img["id"])

            selected_sar_annotations = []
            for ann in sar_annotations:
                if ann["image_id"] in selected_sar_ids:
                    selected_sar_annotations.append(ann)

            # Offset SAR image IDs to avoid collision with optical IDs
            max_opt_id = max(img["id"] for img in opt_images)
            id_offset = max_opt_id + 1

            merged_images = deepcopy(opt_images)
            merged_annotations = deepcopy(opt_annotations)

            for img in selected_sar_images:
                new_img = deepcopy(img)
                new_img["id"] += id_offset
                merged_images.append(new_img)

            max_ann_id = max((ann["id"] for ann in opt_annotations), default=0)
            ann_id_offset = max_ann_id + 1

            for ann in selected_sar_annotations:
                new_ann = deepcopy(ann)
                new_ann["id"] += ann_id_offset
                new_ann["image_id"] += id_offset
                merged_annotations.append(new_ann)

            concat_dir = os.path.join(output_dir, "optical_ssdd", "annotations")
            save_json(
                {"images": merged_images, "annotations": merged_annotations,
                 "categories": optical_anno["categories"]},
                os.path.join(
                    concat_dir,
                    f"instances_train2017.{fold}@{shot}_dior_annotations.json"
                ),
            )

            print(f"  Fold {fold}, {shot}-shot: "
                  f"{len(merged_images)} images "
                  f"({len(opt_images)} optical + {len(selected_sar_images)} SAR), "
                  f"{len(merged_annotations)} annotations")


def create_symlinks(optical_img_dir, sar_img_dir, output_dir):
    """Step 5: Create symbolic links for image directories."""
    print("\n[Step 5] Creating symbolic links for image directories...")

    optical_link = os.path.join(output_dir, "optical", "images")
    if not os.path.exists(optical_link):
        os.makedirs(os.path.dirname(optical_link), exist_ok=True)
        try:
            os.symlink(os.path.abspath(optical_img_dir), optical_link)
            print(f"  {optical_link} -> {optical_img_dir}")
        except OSError:
            print(f"  Symlink failed. Please manually copy/link: {optical_img_dir} -> {optical_link}")

    sar_link = os.path.join(output_dir, "ssdd", "JPEGImages")
    if not os.path.exists(sar_link):
        os.makedirs(os.path.dirname(sar_link), exist_ok=True)
        try:
            os.symlink(os.path.abspath(sar_img_dir), sar_link)
            print(f"  {sar_link} -> {sar_img_dir}")
        except OSError:
            print(f"  Symlink failed. Please manually copy/link: {sar_img_dir} -> {sar_link}")

    # Also link SAR images under optical_ssdd for pretrain phase 2
    sar_link2 = os.path.join(output_dir, "optical_ssdd", "images")
    if not os.path.exists(sar_link2):
        os.makedirs(os.path.dirname(sar_link2), exist_ok=True)
        # For Do ∪ Dl, we need both optical and SAR images accessible.
        # Simplest approach: symlink to a parent that contains both.
        # The config will handle separate paths.
        print(f"  NOTE: Do ∪ Dl uses separate image paths for optical and SAR.")
        print(f"  Optical: {optical_img_dir}")
        print(f"  SAR: {sar_img_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Dual Teacher reproduction")
    parser.add_argument("--optical-ann", type=str, required=True,
                        help="Path to ShipRSImageNet annotation file (COCO format)")
    parser.add_argument("--optical-img", type=str, required=True,
                        help="Path to ShipRSImageNet image directory")
    parser.add_argument("--sar-ann", type=str, required=True,
                        help="Path to SSDD training annotation file (train.json)")
    parser.add_argument("--sar-img", type=str, required=True,
                        help="Path to SSDD image directory (JPEGImages/)")
    parser.add_argument("--sar-test-ann", type=str, default=None,
                        help="Path to SSDD test annotation file")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Output data directory")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 3, 5, 10],
                        help="Number of labeled SAR images for few-shot")
    parser.add_argument("--num-folds", type=int, default=5,
                        help="Number of random folds per shot count")
    args = parser.parse_args()

    print("=" * 60)
    print("Dual Teacher Data Preparation")
    print("  Optical: ShipRSImageNet")
    print("  SAR: SSDD")
    print(f"  Shots: {args.shots}")
    print(f"  Folds: {args.num_folds}")
    print("=" * 60)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Optical annotations
    optical_anno = prepare_optical_annotations(args.optical_ann, output_dir)

    # Step 2: SSDD few-shot splits
    prepare_ssdd_fewshot(args.sar_ann, output_dir, args.shots, args.num_folds)

    # Step 3: SAR test
    prepare_sar_test(args.sar_test_ann, output_dir)

    # Step 4: Do ∪ Dl concatenation
    prepare_do_union_dl(optical_anno, args.sar_ann, output_dir, args.shots, args.num_folds)

    # Step 5: Symlinks
    create_symlinks(args.optical_img, args.sar_img, output_dir)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"    optical/")
    print(f"      dior_annotations.json       # ShipRSImageNet (sup1)")
    print(f"      images/                      # -> {args.optical_img}")
    print(f"    ssdd/")
    print(f"      annotations/")
    print(f"        train.json                 # Full SSDD train")
    print(f"        test.json                  # SSDD test")
    print(f"        semi_supervised/")
    print(f"          instances_train2017.{{fold}}@{{shot}}.json           # labeled SAR")
    print(f"          instances_train2017.{{fold}}@{{shot}}-unlabeled.json  # unlabeled SAR")
    print(f"      JPEGImages/                  # -> {args.sar_img}")
    print(f"    optical_ssdd/")
    print(f"      annotations/")
    print(f"        instances_train2017.{{fold}}@{{shot}}_dior_annotations.json  # Do ∪ Dl")
    print("=" * 60)


if __name__ == "__main__":
    main()
