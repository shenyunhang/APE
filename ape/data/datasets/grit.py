import os

from .coco import custom_register_coco_instances

GRIT_CATEGORIES = [
    {"id": 0, "name": "object"},
]


def _get_builtin_metadata(dataset_name):
    id_to_name = {x["id"]: x["name"] for x in GRIT_CATEGORIES}
    thing_dataset_id_to_contiguous_id = {i: i for i in range(len(GRIT_CATEGORIES))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }


_PREDEFINED_SPLITS_GRIT = {
    "grit": ("GRIT/images", "GRIT/grit.json"),
    "grit_0_snappy": ("GRIT/images", "GRIT/grit_0_snappy.json"),
    "grit_1_snappy": ("GRIT/images", "GRIT/grit_1_snappy.json"),
    "grit_2_snappy": ("GRIT/images", "GRIT/grit_2_snappy.json"),
    "grit_3_snappy": ("GRIT/images", "GRIT/grit_3_snappy.json"),
    "grit_4_snappy": ("GRIT/images", "GRIT/grit_4_snappy.json"),
    "grit_5_snappy": ("GRIT/images", "GRIT/grit_5_snappy.json"),
    "grit_6_snappy": ("GRIT/images", "GRIT/grit_6_snappy.json"),
    "grit_7_snappy": ("GRIT/images", "GRIT/grit_7_snappy.json"),
    "grit_8_snappy": ("GRIT/images", "GRIT/grit_8_snappy.json"),
    "grit_9_snappy": ("GRIT/images", "GRIT/grit_9_snappy.json"),
    "grit_10_snappy": ("GRIT/images", "GRIT/grit_10_snappy.json"),
    "grit_11_snappy": ("GRIT/images", "GRIT/grit_11_snappy.json"),
    "grit_12_snappy": ("GRIT/images", "GRIT/grit_12_snappy.json"),
    "grit_13_snappy": ("GRIT/images", "GRIT/grit_13_snappy.json"),
    "grit_14_snappy": ("GRIT/images", "GRIT/grit_14_snappy.json"),
    "grit_15_snappy": ("GRIT/images", "GRIT/grit_15_snappy.json"),
    "grit_16_snappy": ("GRIT/images", "GRIT/grit_16_snappy.json"),
    "grit_17_snappy": ("GRIT/images", "GRIT/grit_17_snappy.json"),
    "grit_18_snappy": ("GRIT/images", "GRIT/grit_18_snappy.json"),
    "grit_19_snappy": ("GRIT/images", "GRIT/grit_19_snappy.json"),
    "grit_20_snappy": ("GRIT/images", "GRIT/grit_20_snappy.json"),
    "grit_21_snappy": ("GRIT/images", "GRIT/grit_21_snappy.json"),
}


def register_all_GRIT(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_GRIT.items():
        custom_register_coco_instances(
            key,
            _get_builtin_metadata(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".grit"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_GRIT(_root)
