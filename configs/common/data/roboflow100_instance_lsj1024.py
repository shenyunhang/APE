import json
import os

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import DatasetMapper, build_detection_test_loader, get_detection_dataset_dicts
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from omegaconf import OmegaConf

dataloader = OmegaConf.create()

data_root = "datasets/rf100"

rf100_dataset_names = []
for root, dirs, files in os.walk(data_root):
    for d in dirs:
        if root == data_root:
            pass
        else:
            continue

        rf100_dataset_names.append(d)

        d = os.path.join(root, d)
        print(len(rf100_dataset_names), d)
print(rf100_dataset_names, len(rf100_dataset_names))
assert len(rf100_dataset_names) == 100


def _get_builtin_metadata(name):
    meta = {}
    json_file = os.path.join(data_root, name, "valid", "_annotations.coco.json")
    with open(json_file, "r") as fr:
        json_data = json.load(fr)
    meta["thing_classes"] = [category["name"] for category in json_data["categories"]]

    return meta


for key in rf100_dataset_names:
    print("register_coco_instances", key)
    register_coco_instances(
        "rf100_" + key,
        _get_builtin_metadata(key),
        os.path.join(data_root, key, "valid", "_annotations.coco.json"),
        os.path.join(data_root, key, "valid"),
    )

dataloader.tests = [
    L(build_detection_test_loader)(
        dataset=L(get_detection_dataset_dicts)(names="rf100_" + name, filter_empty=False),
        mapper=L(DatasetMapper)(
            is_train=False,
            augmentations=[
                L(T.ResizeShortestEdge)(short_edge_length=1024, max_size=1024),
            ],
            image_format="RGB",
        ),
        num_workers=4,
    )
    for name in rf100_dataset_names
]

dataloader.name_prompt_fusion_text = [True] * len(rf100_dataset_names)


dataloader.select_box_nums_for_evaluation_list = [300] * len(rf100_dataset_names)

dataloader.evaluators = [
    L(COCOEvaluator)(
        dataset_name="rf100_" + name,
        tasks=("bbox",),
    )
    for name in rf100_dataset_names
]
