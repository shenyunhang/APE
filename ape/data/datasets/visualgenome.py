import logging
import os

from .coco import custom_register_coco_instances
from .visualgenome_categories import (
    VISUALGENOME_150_CATEGORIES,
    VISUALGENOME_1356_CATEGORIES,
    VISUALGENOME_1356MINUS150_CATEGORIES,
    VISUALGENOME_77962_CATEGORIES,
    VISUALGENOME_77962MINUS150_CATEGORIES,
)

logger = logging.getLogger(__name__)


def _get_builtin_metadata(dataset_name):
    if dataset_name == "visualgenome_150_box":
        return _get_visualgenome_metadata(VISUALGENOME_150_CATEGORIES)

    if dataset_name == "visualgenome_region":
        return _get_visualgenome_metadata([])

    if dataset_name == "visualgenome_150_box_and_region":
        return _get_visualgenome_metadata(VISUALGENOME_150_CATEGORIES)

    if dataset_name == "visualgenome_77962_box_and_region":
        return _get_visualgenome_metadata(VISUALGENOME_77962_CATEGORIES)

    if dataset_name == "visualgenome_77962_box":
        return _get_visualgenome_metadata(VISUALGENOME_77962_CATEGORIES)

    if dataset_name == "visualgenome_77962minus150_box":
        return _get_visualgenome_metadata(VISUALGENOME_77962MINUS150_CATEGORIES)

    if dataset_name == "visualgenome_77962minus2319_box":
        return _get_visualgenome_metadata(VISUALGENOME_77962MINUS150_CATEGORIES)

    if dataset_name == "visualgenome_1356_box":
        return _get_visualgenome_metadata(VISUALGENOME_1356_CATEGORIES)

    if dataset_name == "visualgenome_1356minus150_box":
        return _get_visualgenome_metadata(VISUALGENOME_1356MINUS150_CATEGORIES)

    if dataset_name == "visualgenome_1356minus2319_box":
        return _get_visualgenome_metadata(VISUALGENOME_1356MINUS150_CATEGORIES)

    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))


def _get_visualgenome_metadata(categories):
    if len(categories) == 0:
        return {}
    id_to_name = {x["id"]: x["name"] for x in categories}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(categories))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }


_PREDEFINED_SPLITS_VISUALGENOME = {}
_PREDEFINED_SPLITS_VISUALGENOME["visualgenome_150_box"] = {
    "visualgenome_150_box_train": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_150_box_train.json",
    ),
    "visualgenome_150_box_val": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_150_box_val.json",
    ),
}

_PREDEFINED_SPLITS_VISUALGENOME["visualgenome_150_box_and_region"] = {
    "visualgenome_150_box_and_region_train": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_150_box_and_region_train.json",
    ),
    "visualgenome_150_box_and_region_val": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_150_box_and_region_val.json",
    ),
}

_PREDEFINED_SPLITS_VISUALGENOME["visualgenome_77962_box"] = {
    "visualgenome_77962_box": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_77962_box.json",
    ),
    "visualgenome_77962_box_train": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_77962_box_train.json",
    ),
    "visualgenome_77962_box_val": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_77962_box_val.json",
    ),
}

_PREDEFINED_SPLITS_VISUALGENOME["visualgenome_77962_box_and_region"] = {
    "visualgenome_77962_box_and_region": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_77962_box_and_region.json",
    ),
    "visualgenome_77962_box_and_region_train": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_77962_box_and_region_train.json",
    ),
    "visualgenome_77962_box_and_region_val": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_77962_box_and_region_val.json",
    ),
}

_PREDEFINED_SPLITS_VISUALGENOME["visualgenome_region"] = {
    "visualgenome_region": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_region.json",
    ),
    "visualgenome_region_train": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_region_train.json",
    ),
    "visualgenome_region_val": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_region_val.json",
    ),
}

_PREDEFINED_SPLITS_VISUALGENOME["visualgenome_77962minus150_box"] = {
    "visualgenome_77962minus150_box": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_77962minus150_box.json",
    ),
    "visualgenome_77962minus150_box_train": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_77962minus150_box_train.json",
    ),
    "visualgenome_77962minus150_box_val": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_77962minus150_box_val.json",
    ),
}

_PREDEFINED_SPLITS_VISUALGENOME["visualgenome_77962minus2319_box"] = {
    "visualgenome_77962minus2319_box": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_77962minus2319_box.json",
    ),
    "visualgenome_77962minus2319_box_train": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_77962minus2319_box_train.json",
    ),
    "visualgenome_77962minus2319_box_val": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_77962minus2319_box_val.json",
    ),
}

_PREDEFINED_SPLITS_VISUALGENOME["visualgenome_1356_box"] = {
    "visualgenome_1356_box": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_1356_box.json",
    ),
    "visualgenome_1356_box_train": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_1356_box_train.json",
    ),
    "visualgenome_1356_box_val": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_1356_box_val.json",
    ),
}

_PREDEFINED_SPLITS_VISUALGENOME["visualgenome_1356minus150_box"] = {
    "visualgenome_1356minus150_box": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_1356minus150_box.json",
    ),
    "visualgenome_1356minus150_box_train": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_1356minus150_box_train.json",
    ),
    "visualgenome_1356minus150_box_val": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_1356minus150_box_val.json",
    ),
}

_PREDEFINED_SPLITS_VISUALGENOME["visualgenome_1356minus2319_box"] = {
    "visualgenome_1356minus2319_box": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_1356minus2319_box.json",
    ),
    "visualgenome_1356minus2319_box_train": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_1356minus2319_box_train.json",
    ),
    "visualgenome_1356minus2319_box_val": (
        "visualgenome",
        "visualgenome/annotations/visualgenome_1356minus2319_box_val.json",
    ),
}


def register_all_visualgenome(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_VISUALGENOME.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            custom_register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


if __name__.endswith(".visualgenome"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_all_visualgenome(_root)
