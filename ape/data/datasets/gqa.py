import logging
import os

from .coco import custom_register_coco_instances

logger = logging.getLogger(__name__)


def _get_builtin_metadata(dataset_name):
    return _get_gqa_metadata([])

    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))


def _get_gqa_metadata(categories):
    if len(categories) == 0:
        return {}
    id_to_name = {x["id"]: x["name"] for x in categories}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(categories))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }


_PREDEFINED_SPLITS_GQA = {}
_PREDEFINED_SPLITS_GQA["gqa_region"] = {
    "gqa_region": (
        "gqa/images",
        "gqa/gqa_region.json",
    ),
    "gqa_region_train": (
        "gqa/images",
        "gqa/gqa_region_train.json",
    ),
    "gqa_region_val": (
        "gqa/images",
        "gqa/gqa_region_val.json",
    ),
}


def register_all_gqa(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_GQA.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            custom_register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".gqa"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_all_gqa(_root)
