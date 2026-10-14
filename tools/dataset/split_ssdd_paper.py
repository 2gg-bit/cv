#!/usr/bin/env python
"""Create the SSDD train/test split specified by the Dual Teacher paper.

The paper assigns SSDD images whose numeric index ends in 1 or 9 to the test
set and uses all remaining images for training. This script accepts one full
COCO annotation file and writes disjoint ``train.json`` and ``test.json``
files without changing image IDs or annotation records.

For the canonical 1,160-image SSDD release, the expected result is 928 train
images and 232 test images. Noncanonical counts are rejected unless the caller
explicitly opts in (useful for synthetic tests or documented dataset variants).
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import tempfile


PAPER_TEST_SUFFIXES = (1, 9)
CANONICAL_IMAGE_COUNT = 1160
CANONICAL_TEST_COUNT = 232


def _numeric_image_index(image):
    """Return the SSDD image index, rejecting ambiguous COCO identities."""
    if not isinstance(image, dict):
        raise ValueError("Each COCO image entry must be an object")
    image_id = image.get("id")
    if type(image_id) is not int or image_id < 0:
        raise ValueError("SSDD image id must be a nonnegative integer: {!r}".format(image_id))

    file_name = image.get("file_name")
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("SSDD image file_name must be a nonempty string")
    basename = os.path.basename(file_name.replace("\\", "/"))
    stem = os.path.splitext(basename)[0]
    if not stem.isdigit():
        raise ValueError(
            "SSDD file_name must have a numeric image-index stem: {!r}".format(file_name)
        )
    image_index = int(stem)
    if image_index != image_id:
        raise ValueError(
            "SSDD COCO id must match the numeric file index: id={!r}, file_name={!r}".format(
                image_id, file_name
            )
        )
    return image_index


def partition_ssdd_coco(coco, require_canonical_counts=True):
    """Partition a full SSDD COCO object using the paper's suffix rule."""
    if not isinstance(coco, dict):
        raise ValueError("Input COCO document must be an object")
    if not isinstance(coco.get("images"), list):
        raise ValueError("Input COCO document must contain an images list")
    if not isinstance(coco.get("annotations"), list):
        raise ValueError("Input COCO document must contain an annotations list")

    image_by_id = {}
    train_images = []
    test_images = []
    for image in coco["images"]:
        image_index = _numeric_image_index(image)
        image_id = image["id"]
        if image_id in image_by_id:
            raise ValueError("Duplicate SSDD image id: {}".format(image_id))
        image_by_id[image_id] = image
        if image_index % 10 in PAPER_TEST_SUFFIXES:
            test_images.append(image)
        else:
            train_images.append(image)

    train_ids = set(image["id"] for image in train_images)
    test_ids = set(image["id"] for image in test_images)
    if train_ids.intersection(test_ids):
        raise ValueError("Train/test image IDs overlap")
    if train_ids.union(test_ids) != set(image_by_id):
        raise ValueError("Train/test partition does not cover the input images")

    train_annotations = []
    test_annotations = []
    for annotation in coco["annotations"]:
        if not isinstance(annotation, dict) or "image_id" not in annotation:
            raise ValueError("Each COCO annotation must contain image_id")
        image_id = annotation["image_id"]
        if image_id in train_ids:
            train_annotations.append(annotation)
        elif image_id in test_ids:
            test_annotations.append(annotation)
        else:
            raise ValueError(
                "Annotation references an image absent from the input: {!r}".format(image_id)
            )

    if require_canonical_counts:
        if len(image_by_id) != CANONICAL_IMAGE_COUNT:
            raise ValueError(
                "Expected the canonical {}-image SSDD release, found {} images. "
                "Use --allow-noncanonical-counts only for an intentional variant.".format(
                    CANONICAL_IMAGE_COUNT, len(image_by_id)
                )
            )
        expected_ids = set(range(1, CANONICAL_IMAGE_COUNT + 1))
        if set(image_by_id) != expected_ids:
            missing = sorted(expected_ids.difference(image_by_id))
            unexpected = sorted(set(image_by_id).difference(expected_ids))
            raise ValueError(
                "Canonical SSDD indices must be 1..1160; missing={}, unexpected={}".format(
                    missing[:10], unexpected[:10]
                )
            )
        if len(test_images) != CANONICAL_TEST_COUNT:
            raise ValueError(
                "Paper suffix rule should produce {} test images from canonical SSDD, "
                "found {}.".format(CANONICAL_TEST_COUNT, len(test_images))
            )

    common = dict((key, value) for key, value in coco.items()
                  if key not in ("images", "annotations"))
    train = dict(common)
    train["images"] = train_images
    train["annotations"] = train_annotations
    test = dict(common)
    test["images"] = test_images
    test["annotations"] = test_annotations

    return train, test


def validate_ssdd_training_annotations(coco):
    """Reject a training annotation file containing paper-test images."""
    if not isinstance(coco, dict) or not isinstance(coco.get("images"), list):
        raise ValueError("SSDD training COCO document must contain an images list")
    if not isinstance(coco.get("annotations", []), list):
        raise ValueError("SSDD training COCO annotations must be a list")
    seen = set()
    for image in coco["images"]:
        image_index = _numeric_image_index(image)
        image_id = image["id"]
        if image_id in seen:
            raise ValueError("Duplicate SSDD image id: {}".format(image_id))
        seen.add(image_id)
        if image_index % 10 in PAPER_TEST_SUFFIXES:
            raise ValueError(
                "Paper test image {} ({}) is present in the SSDD training split".format(
                    image_id, image.get("file_name")
                )
            )
    for annotation in coco.get("annotations", []):
        if not isinstance(annotation, dict) or annotation.get("image_id") not in seen:
            raise ValueError("SSDD training annotation references an unknown image")
    return len(seen)


def validate_ssdd_paper_split(train_coco, test_coco, require_canonical_counts=False):
    """Validate an existing train/test pair against the paper's suffix rule."""
    train_count = validate_ssdd_training_annotations(train_coco)
    if not isinstance(test_coco, dict) or not isinstance(test_coco.get("images"), list):
        raise ValueError("SSDD test COCO document must contain an images list")
    if not isinstance(test_coco.get("annotations", []), list):
        raise ValueError("SSDD test COCO annotations must be a list")

    train_ids = set(image["id"] for image in train_coco["images"])
    test_ids = set()
    for image in test_coco["images"]:
        image_index = _numeric_image_index(image)
        image_id = image["id"]
        if image_id in train_ids or image_id in test_ids:
            raise ValueError("SSDD train/test image IDs overlap or repeat: {}".format(image_id))
        if image_index % 10 not in PAPER_TEST_SUFFIXES:
            raise ValueError(
                "Non-test SSDD image {} ({}) is present in the test split".format(
                    image_id, image.get("file_name")
                )
            )
        test_ids.add(image_id)

    for annotation in test_coco.get("annotations", []):
        if not isinstance(annotation, dict) or annotation.get("image_id") not in test_ids:
            raise ValueError("SSDD test annotation references an unknown image")

    if require_canonical_counts:
        if train_count + len(test_ids) != CANONICAL_IMAGE_COUNT:
            raise ValueError("Expected 1160 images across the canonical SSDD split")
        if train_ids.union(test_ids) != set(range(1, CANONICAL_IMAGE_COUNT + 1)):
            raise ValueError("Canonical SSDD split must cover image indices 1..1160 exactly")
        if len(test_ids) != CANONICAL_TEST_COUNT:
            raise ValueError("Expected 232 SSDD test images under the paper split")
    return {"images_train": train_count, "images_test": len(test_ids)}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _write_json_temp(directory, prefix, value):
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
            stream.write(payload)
            stream.write(b"\n")
        return path, _sha256(payload + b"\n")
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def split_file(input_json, output_dir, overwrite=False, require_canonical_counts=True):
    """Split a source COCO JSON and write train/test files plus a manifest."""
    input_path = os.path.abspath(input_json)
    output_dir = os.path.abspath(output_dir)
    with open(input_path, "rb") as stream:
        source_bytes = stream.read()
    try:
        coco = json.loads(source_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("Could not parse input COCO JSON: {}".format(exc))

    train, test = partition_ssdd_coco(coco, require_canonical_counts)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_paths = {
        "train": os.path.join(output_dir, "train.json"),
        "test": os.path.join(output_dir, "test.json"),
        "manifest": os.path.join(output_dir, "paper_split_manifest.json"),
    }
    if input_path in set(os.path.abspath(path) for path in output_paths.values()):
        raise ValueError("Input JSON cannot be overwritten by a split output")
    existing = [path for path in output_paths.values() if os.path.exists(path)]
    if existing and not overwrite:
        raise ValueError(
            "Refusing to overwrite existing output(s): {}. Pass --overwrite explicitly.".format(
                ", ".join(existing)
            )
        )

    temp_paths = []
    try:
        train_tmp, train_hash = _write_json_temp(output_dir, ".train.", train)
        temp_paths.append(train_tmp)
        test_tmp, test_hash = _write_json_temp(output_dir, ".test.", test)
        temp_paths.append(test_tmp)

        manifest = {
            "protocol": "Dual Teacher (2023) SSDD split",
            "source_json": input_path,
            "source_sha256": _sha256(source_bytes),
            "rule": "numeric image index ending in 1 or 9 -> test; all others -> train",
            "test_suffixes": list(PAPER_TEST_SUFFIXES),
            "counts": {
                "images_total": len(coco["images"]),
                "images_train": len(train["images"]),
                "images_test": len(test["images"]),
                "annotations_total": len(coco["annotations"]),
                "annotations_train": len(train["annotations"]),
                "annotations_test": len(test["annotations"]),
            },
            "output_sha256": {"train.json": train_hash, "test.json": test_hash},
            "canonical_expected": {
                "images_total": CANONICAL_IMAGE_COUNT,
                "images_test": CANONICAL_TEST_COUNT,
                "strict_count_check_enabled": bool(require_canonical_counts),
            },
        }
        manifest_tmp, _ = _write_json_temp(output_dir, ".paper_split_manifest.", manifest)
        temp_paths.append(manifest_tmp)

        # All content is fully written and validated before any destination changes.
        for key, temp_path in zip(("train", "test", "manifest"), temp_paths[:]):
            os.replace(temp_path, output_paths[key])
            temp_paths.remove(temp_path)
    finally:
        for path in temp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass

    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True,
                        help="One COCO annotation JSON containing the full SSDD image set")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for train.json, test.json and the split manifest")
    parser.add_argument("--allow-noncanonical-counts", action="store_true",
                        help="Allow a dataset variant instead of requiring 1160/232 images")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing train.json/test.json/manifest in output-dir")
    args = parser.parse_args()

    try:
        manifest = split_file(
            args.input_json,
            args.output_dir,
            overwrite=args.overwrite,
            require_canonical_counts=not args.allow_noncanonical_counts,
        )
    except (IOError, OSError, ValueError) as exc:
        parser.exit(2, "error: {}\n".format(exc))

    counts = manifest["counts"]
    print("SSDD paper split complete:")
    print("  train: {} images / {} annotations".format(
        counts["images_train"], counts["annotations_train"]))
    print("  test:  {} images / {} annotations".format(
        counts["images_test"], counts["annotations_test"]))
    print("  manifest: {}".format(os.path.join(args.output_dir, "paper_split_manifest.json")))


if __name__ == "__main__":
    main()
