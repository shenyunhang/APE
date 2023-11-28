import os

from detectron2.data.datasets.register_coco import register_coco_instances

SA1B_CATEGORIES = [
    {"id": 1, "name": "object"},
]


def _get_builtin_metadata(key):
    id_to_name = {x["id"]: x["name"] for x in SA1B_CATEGORIES}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(SA1B_CATEGORIES))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }


_PREDEFINED_SPLITS_SA1B = {
    "sa1b": ("SA-1B/images", "SA-1B/sam1b_instance.json"),
    "sa1b_1m": ("SA-1B/images", "SA-1B/sam1b_instance_1000000.json"),
    "sa1b_2m": ("SA-1B/images", "SA-1B/sam1b_instance_2000000.json"),
    "sa1b_4m": ("SA-1B/images", "SA-1B/sam1b_instance_4000000.json"),
    "sa1b_6m": ("SA-1B/images", "SA-1B/sam1b_instance_6000000.json"),
    "sa1b_8m": ("SA-1B/images", "SA-1B/sam1b_instance_8000000.json"),
    "sa1b_10m": ("SA-1B/images", "SA-1B/sam1b_instance_10000000.json"),
}


def register_all_sa1b(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_SA1B.items():
        register_coco_instances(
            key,
            _get_builtin_metadata(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".sa1b"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_sa1b(_root)
