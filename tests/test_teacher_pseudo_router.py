import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "teacher_pseudo_router_test", ROOT / "ssod/models/teacher_pseudo_router.py"
)
ROUTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUTER)


def test_agreeing_pair_is_fused_and_score_is_not_boosted():
    result = ROUTER.fuse_teacher_detections(
        [[0, 0, 10, 10, 0.8]], [0],
        [[1, 0, 11, 10, 0.5]], [0],
        iou_threshold=0.8,
    )
    assert result["matched_pairs"] == 1
    assert result["teacher1_only"] == result["teacher2_only"] == 0
    assert result["boxes"].shape == (1, 5)
    assert result["labels"].tolist() == [0]
    assert result["sources"].tolist() == [3]
    assert result["boxes"][0, 0] == pytest.approx(0.5 / 1.3)
    assert result["boxes"][0, 2] == pytest.approx(13.5 / 1.3)
    assert result["boxes"][0, 4] == pytest.approx(np.sqrt(0.8 * 0.5))
    assert result["boxes"][0, 4] <= max(0.8, 0.5)


def test_single_teacher_candidates_are_retained_with_source_ids():
    result = ROUTER.fuse_teacher_detections(
        [[0, 0, 10, 10, 0.8]], [0],
        [[20, 20, 30, 30, 0.7]], [0],
    )
    assert result["matched_pairs"] == 0
    assert result["teacher1_only"] == result["teacher2_only"] == 1
    assert result["sources"].tolist() == [1, 2]
    assert result["boxes"][:, 4].tolist() == pytest.approx([0.8, 0.7])


def test_class_mismatch_does_not_fuse_overlapping_boxes():
    result = ROUTER.fuse_teacher_detections(
        [[0, 0, 10, 10, 0.8]], [0],
        [[0, 0, 10, 10, 0.7]], [1],
    )
    assert result["matched_pairs"] == 0
    assert len(result["boxes"]) == 2
    assert sorted(result["sources"].tolist()) == [1, 2]


def test_empty_teacher_is_a_passthrough():
    result = ROUTER.fuse_teacher_detections(
        np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64),
        [[2, 3, 8, 9, 0.75]], [0],
    )
    np.testing.assert_array_equal(result["boxes"], np.asarray([[2, 3, 8, 9, 0.75]], dtype=np.float32))
    assert result["sources"].tolist() == [2]


def test_greedy_matching_is_one_to_one_and_deterministic():
    left = [[0, 0, 10, 10, 0.9], [1, 0, 11, 10, 0.8]]
    right = [[0, 0, 10, 10, 0.9], [2, 0, 12, 10, 0.8]]
    first = ROUTER.fuse_teacher_detections(left, [0, 0], right, [0, 0], iou_threshold=0.5)
    second = ROUTER.fuse_teacher_detections(left, [0, 0], right, [0, 0], iou_threshold=0.5)
    assert first["matched_pairs"] == 2
    np.testing.assert_array_equal(first["boxes"], second["boxes"])
    np.testing.assert_array_equal(first["sources"], [3, 3])


@pytest.mark.parametrize("threshold", [0, -0.1, 1.01, float("nan")])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="iou_threshold"):
        ROUTER.fuse_teacher_detections([], [], [], [], iou_threshold=threshold)


def test_nonfinite_boxes_rejected():
    with pytest.raises(ValueError, match="NaN or Inf"):
        ROUTER.fuse_teacher_detections([[0, 0, np.inf, 10, 0.8]], [0], [], [])
