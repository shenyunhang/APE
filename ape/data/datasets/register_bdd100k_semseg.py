# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import os
from typing import List, Tuple, Union

import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

BDD_SEM = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

__all__ = ["load_scannet_instances", "register_scannet_context"]


def load_bdd_instances(
    name: str, dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]
):
    """
    Load BDD annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    img_folder = os.path.join(dirname, "images", "10k", split)
    img_pths = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))

    sem_folder = os.path.join(dirname, "labels", "sem_seg", "masks", split)
    sem_pths = sorted(glob.glob(os.path.join(sem_folder, "*.png")))

    assert len(img_pths) == len(sem_pths)

    dicts = []
    for img_pth, sem_pth in zip(img_pths, sem_pths):
        r = {
            "file_name": img_pth,
            "sem_seg_file_name": sem_pth,
            "image_id": img_pth.split("/")[-1].split(".")[0],
        }
        dicts.append(r)
    return dicts


def register_bdd_context(name, dirname, split, class_names=BDD_SEM):
    DatasetCatalog.register(name, lambda: load_bdd_instances(name, dirname, split, class_names))
    MetadataCatalog.get(name).set(
        stuff_classes=class_names,
        dirname=dirname,
        split=split,
        ignore_label=[255],
        thing_dataset_id_to_contiguous_id={},
        class_offset=0,
        keep_sem_bgd=False,
    )


def register_all_bdd_semseg(root):
    SPLITS = [
        ("bdd10k_val_sem_seg", "bdd100k", "val"),
    ]

    for name, dirname, split in SPLITS:
        register_bdd_context(name, os.path.join(root, dirname), split)
        MetadataCatalog.get(name).evaluator_type = "sem_seg"


if __name__.endswith(".register_bdd100k_semseg"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_bdd_semseg(_root)
