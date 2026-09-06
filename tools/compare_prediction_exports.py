"""Compare complete COCO prediction exports without MMDetection or CUDA.

Usage: python tools/compare_prediction_exports.py EXPORT_A EXPORT_B

Default comparison is exact and insensitive to prediction/image ordering. An
explicit --bbox-atol (original pixels) or --score-atol (probability units) permits
absolute tolerance, using one-to-one matching, not greedy nearest neighbours.
Image manifests and annotation-file SHA256 must agree. Empty images count through
the manifest, not through prediction rows. Exit 0 means equal predictions, 1 means
different predictions, and 2 means invalid/incomplete exports. Metric differences
are reported separately; rounded AP equality alone never proves equivalence.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    def invalid_constant(value):
        raise ValueError("non-finite JSON constant: {}".format(value))
    with open(str(path), encoding="utf-8") as handle:
        return json.load(handle, parse_constant=invalid_constant)


def file_hash(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_predictions(predictions, image_ids, category_ids):
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a COCO result list")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("duplicate manifest image IDs")
    image_ids, category_ids = set(image_ids), set(category_ids)
    grouped = defaultdict(list)
    for index, item in enumerate(predictions):
        if not isinstance(item, dict):
            raise ValueError("prediction {} is not an object".format(index))
        image_id, category_id = item.get("image_id"), item.get("category_id")
        if type(image_id) is not int or image_id not in image_ids:
            raise ValueError("prediction {} has unknown image_id".format(index))
        if type(category_id) is not int or category_id not in category_ids:
            raise ValueError("prediction {} has unknown category_id".format(index))
        bbox, score = item.get("bbox"), item.get("score")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(_finite_number(v) for v in bbox):
            raise ValueError("prediction {} has invalid/non-finite xywh bbox".format(index))
        if bbox[2] < 0 or bbox[3] < 0:
            raise ValueError("prediction {} has negative width/height".format(index))
        if not _finite_number(score) or not 0 <= score <= 1:
            raise ValueError("prediction {} has invalid/non-finite score".format(index))
        grouped[(image_id, category_id)].append(tuple(bbox) + (score,))
    return grouped


def _annotation_path(directory, metadata, explicit_annotation):
    candidates = []
    if metadata.get("test_ann_snapshot"):
        candidates.append(directory / metadata["test_ann_snapshot"])
    if explicit_annotation:
        candidates.append(Path(explicit_annotation))
    candidates.extend([directory / "test.json", directory.parent / "test.json"])
    if metadata.get("test_ann_file"):
        candidates.append(Path(metadata["test_ann_file"]))
    for path in candidates:
        if path.is_file():
            return path
    raise ValueError("no test annotation available for {}; use --annotations".format(directory))


def load_export(directory, annotation_path=None, expected_count=232):
    directory = Path(directory)
    metadata = read_json(directory / "metadata.json")
    image_ids = metadata.get("image_ids")
    if not isinstance(image_ids, list) or any(type(value) is not int for value in image_ids):
        raise ValueError("metadata.image_ids must be an integer list")
    if len(image_ids) != expected_count or len(set(image_ids)) != expected_count:
        raise ValueError("metadata must cover {} unique test images".format(expected_count))
    if metadata.get("num_test_images") != expected_count:
        raise ValueError("num_test_images disagrees with the required test manifest")
    annotation_file = _annotation_path(directory, metadata, annotation_path)
    annotation = read_json(annotation_file)
    true_ids = [item["id"] for item in annotation["images"]]
    if (len(true_ids) != expected_count or len(set(true_ids)) != expected_count
            or any(type(value) is not int for value in true_ids)
            or set(true_ids) != set(image_ids)):
        raise ValueError("manifest does not cover exactly the test annotation image IDs")
    category_ids = [item["id"] for item in annotation["categories"]]
    if (not category_ids or any(type(value) is not int for value in category_ids)
            or len(set(category_ids)) != len(category_ids)):
        raise ValueError("annotation has invalid category IDs")
    if "category_ids" in metadata and set(metadata["category_ids"]) != set(category_ids):
        raise ValueError("metadata category IDs disagree with the annotation")
    annotation_sha256 = file_hash(annotation_file)
    if metadata.get("test_ann_sha256", annotation_sha256) != annotation_sha256:
        raise ValueError("annotation SHA256 does not match metadata")
    predictions = read_json(directory / "predictions.bbox.json")
    grouped = validate_predictions(predictions, image_ids, category_ids)
    if metadata.get("num_predictions", len(predictions)) != len(predictions):
        raise ValueError("prediction count disagrees with metadata")
    empty_ids = set(image_ids) - {item["image_id"] for item in predictions}
    if "empty_prediction_image_ids" in metadata and set(metadata["empty_prediction_image_ids"]) != empty_ids:
        raise ValueError("empty-image manifest disagrees with predictions")
    metrics = read_json(directory / "metrics.json")
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a JSON object")
    for key, value in metrics.items():
        if type(value) in (float, int) and not math.isfinite(value):
            raise ValueError("non-finite metric {}".format(key))
    return dict(directory=str(directory), metadata=metadata, image_ids=set(image_ids),
                annotation_sha256=annotation_sha256, grouped=grouped,
                count=len(predictions), empty_ids=empty_ids, metrics=metrics)


def _matched_rows(rows_a, rows_b, bbox_atol, score_atol):
    if len(rows_a) != len(rows_b):
        return False
    if bbox_atol == 0 and score_atol == 0:
        return Counter(rows_a) == Counter(rows_b)
    # A perfect bipartite matching avoids false mismatches when several close
    # candidates exist; each prediction may match exactly one counterpart.
    edges = [[j for j, b in enumerate(rows_b)
              if all(abs(a[k] - b[k]) <= bbox_atol for k in range(4))
              and abs(a[4] - b[4]) <= score_atol] for a in rows_a]
    assigned = {}

    def augment(index, visited):
        for candidate in edges[index]:
            if candidate in visited:
                continue
            visited.add(candidate)
            if candidate not in assigned or augment(assigned[candidate], visited):
                assigned[candidate] = index
                return True
        return False

    return all(augment(i, set()) for i in range(len(rows_a)))


def compare_exports(left, right, bbox_atol=0.0, score_atol=0.0):
    if not _finite_number(bbox_atol) or not _finite_number(score_atol) or min(bbox_atol, score_atol) < 0:
        raise ValueError("tolerances must be finite non-negative numbers")
    if left["image_ids"] != right["image_ids"]:
        raise ValueError("export image manifests differ")
    if left["annotation_sha256"] != right["annotation_sha256"]:
        raise ValueError("test annotation hashes differ; cannot compare different GT")
    changed_groups = []
    for image_id, category_id in sorted(set(left["grouped"]) | set(right["grouped"])):
        rows_a = left["grouped"].get((image_id, category_id), [])
        rows_b = right["grouped"].get((image_id, category_id), [])
        if not _matched_rows(rows_a, rows_b, bbox_atol, score_atol):
            changed_groups.append(dict(image_id=image_id, category_id=category_id,
                                       count_a=len(rows_a), count_b=len(rows_b)))
    metric_differences = {}
    for key in sorted(set(left["metrics"]) | set(right["metrics"])):
        a, b = left["metrics"].get(key), right["metrics"].get(key)
        if a != b:
            metric_differences[key] = dict(a=a, b=b)
            if _finite_number(a) and _finite_number(b):
                metric_differences[key]["delta_b_minus_a"] = b - a
    return {
        "predictions_equal": not changed_groups,
        "comparison": "order-insensitive one-to-one bbox/category/score",
        "bbox_atol_pixels": bbox_atol, "score_atol": score_atol,
        "num_images": len(left["image_ids"]),
        "test_ann_sha256": left["annotation_sha256"],
        "predictions_a": left["count"], "predictions_b": right["count"],
        "empty_images_a": sorted(left["empty_ids"]),
        "empty_images_b": sorted(right["empty_ids"]),
        "changed_images": sorted({item["image_id"] for item in changed_groups}),
        "changed_groups": changed_groups,
        "metric_differences": metric_differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_a")
    parser.add_argument("export_b")
    parser.add_argument("--annotations", help="fallback common COCO test.json for old exports")
    parser.add_argument("--bbox-atol", type=float, default=0.0, help="absolute original-pixel tolerance")
    parser.add_argument("--score-atol", type=float, default=0.0, help="absolute score tolerance")
    args = parser.parse_args()
    try:
        left = load_export(args.export_a, args.annotations)
        right = load_export(args.export_b, args.annotations)
        report = compare_exports(left, right, args.bbox_atol, args.score_atol)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["predictions_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
