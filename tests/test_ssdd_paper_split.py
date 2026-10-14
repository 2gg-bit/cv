import copy
import hashlib
import json
import os
import tempfile
import unittest

from tools.dataset.split_ssdd_paper import (
    partition_ssdd_coco,
    split_file,
    validate_ssdd_paper_split,
)


def synthetic_coco():
    images = [
        {"id": index, "file_name": "JPEGImages/{:06d}.jpg".format(index),
         "width": 20, "height": 10}
        for index in range(10)
    ]
    annotations = [
        {"id": index + 100, "image_id": index, "category_id": 1,
         "bbox": [0, 0, 2, 2], "area": 4, "iscrowd": 0}
        for index in range(10)
    ]
    return {
        "info": {"source": "synthetic"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "ship"}],
    }


class TestSsddPaperSplit(unittest.TestCase):
    def test_suffixes_one_and_nine_are_test_and_rest_train(self):
        source = synthetic_coco()
        source_before = copy.deepcopy(source)
        train, test = partition_ssdd_coco(source, require_canonical_counts=False)

        self.assertEqual([image["id"] for image in test["images"]], [1, 9])
        self.assertEqual([image["id"] for image in train["images"]],
                         [0, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual([ann["image_id"] for ann in test["annotations"]], [1, 9])
        self.assertEqual([ann["image_id"] for ann in train["annotations"]],
                         [0, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(train["categories"], source["categories"])
        self.assertEqual(test["info"], source["info"])
        self.assertEqual(source, source_before)
        self.assertFalse(set(i["id"] for i in train["images"]) &
                         set(i["id"] for i in test["images"]))

    def test_rejects_filename_id_mismatch(self):
        source = synthetic_coco()
        source["images"][1]["id"] = 101
        with self.assertRaisesRegex(ValueError, "must match"):
            partition_ssdd_coco(source, require_canonical_counts=False)

    def test_rejects_non_numeric_filename(self):
        source = synthetic_coco()
        source["images"][0]["file_name"] = "JPEGImages/ship_a.jpg"
        with self.assertRaisesRegex(ValueError, "numeric image-index"):
            partition_ssdd_coco(source, require_canonical_counts=False)

    def test_rejects_annotation_with_unknown_image(self):
        source = synthetic_coco()
        source["annotations"].append({"id": 999, "image_id": 42})
        with self.assertRaisesRegex(ValueError, "absent from the input"):
            partition_ssdd_coco(source, require_canonical_counts=False)

    def test_rejects_noncanonical_counts_by_default(self):
        with self.assertRaisesRegex(ValueError, "canonical 1160-image"):
            partition_ssdd_coco(synthetic_coco())

    def test_canonical_release_yields_928_train_and_232_test(self):
        source = {
            "images": [
                {"id": index, "file_name": "{:06d}.jpg".format(index)}
                for index in range(1, 1161)
            ],
            "annotations": [],
            "categories": [{"id": 1, "name": "ship"}],
        }
        train, test = partition_ssdd_coco(source)
        self.assertEqual(len(train["images"]), 928)
        self.assertEqual(len(test["images"]), 232)
        self.assertEqual(validate_ssdd_paper_split(train, test, True),
                         {"images_train": 928, "images_test": 232})

        with tempfile.TemporaryDirectory() as temp_dir:
            input_json = os.path.join(temp_dir, "full.json")
            output_dir = os.path.join(temp_dir, "split")
            with open(input_json, "w", encoding="utf-8") as stream:
                json.dump(source, stream)
            manifest = split_file(input_json, output_dir)
            self.assertEqual(manifest["counts"]["images_train"], 928)
            self.assertEqual(manifest["counts"]["images_test"], 232)
            with open(os.path.join(output_dir, "test.json"), "rb") as stream:
                self.assertEqual(hashlib.sha256(stream.read()).hexdigest(),
                                 manifest["output_sha256"]["test.json"])

    def test_existing_split_validator_rejects_test_image_in_training(self):
        source = synthetic_coco()
        train, test = partition_ssdd_coco(source, require_canonical_counts=False)
        contaminated_train = copy.deepcopy(train)
        contaminated_train["images"].append(test["images"][0])
        with self.assertRaisesRegex(ValueError, "Paper test image"):
            validate_ssdd_paper_split(contaminated_train, test)

    def test_writes_manifest_and_refuses_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_json = os.path.join(temp_dir, "all.json")
            output_dir = os.path.join(temp_dir, "split")
            with open(input_json, "w", encoding="utf-8") as stream:
                json.dump(synthetic_coco(), stream)

            manifest = split_file(input_json, output_dir,
                                  require_canonical_counts=False)
            self.assertEqual(manifest["counts"]["images_train"], 8)
            self.assertEqual(manifest["counts"]["images_test"], 2)
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "train.json")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "test.json")))
            self.assertTrue(os.path.isfile(
                os.path.join(output_dir, "paper_split_manifest.json")))
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                split_file(input_json, output_dir,
                           require_canonical_counts=False)


if __name__ == "__main__":
    unittest.main()
