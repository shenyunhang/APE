# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Common data processing utilities that are used in a
typical object detection data pipeline.
"""
import json
import logging
import os
from typing import List, Union

import numpy as np
import pycocotools.mask as mask_util
import torch

from detectron2.data import transforms as T
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.detection_utils import build_augmentation as build_augmentation_d2
from detectron2.data.detection_utils import check_metadata_consistency

from .transforms import AutoAugment, LargeScaleJitter

__all__ = [
    "build_augmentation",
]


def load_fed_loss_cls_weights(class_freq_path: str, freq_weight_power=1.0):
    logger = logging.getLogger(__name__)
    logger.info("Loading " + class_freq_path)
    assert os.path.exists(class_freq_path)

    class_info = json.load(open(class_freq_path, "r"))
    class_freq = torch.tensor([c["image_count"] for c in sorted(class_info, key=lambda x: x["id"])])

    class_freq_weight = class_freq.float() ** freq_weight_power
    return class_freq_weight


def get_fed_loss_cls_weights(dataset_names: Union[str, List[str]], freq_weight_power=1.0):
    """
    Get frequency weight for each class sorted by class id.
    We now calcualte freqency weight using image_count to the power freq_weight_power.

    Args:
        dataset_names: list of dataset names
        freq_weight_power: power value
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    logger = logging.getLogger(__name__)
    class_freq_path = MetadataCatalog.get(dataset_names[0]).json_file[:-5] + "_cat_info.json"
    if os.path.exists(class_freq_path):
        logger.info(
            "Search outside metadata 'image_count' for dataset '{}' from '{}'".format(
                dataset_names[0], class_freq_path
            )
        )
        return load_fed_loss_cls_weights(class_freq_path, freq_weight_power)
    logger.info("Using builtin metadata 'image_count' for dataset '{}'".format(dataset_names))

    check_metadata_consistency("class_image_count", dataset_names)

    meta = MetadataCatalog.get(dataset_names[0])
    class_freq_meta = meta.class_image_count
    class_freq = torch.tensor(
        [c["image_count"] for c in sorted(class_freq_meta, key=lambda x: x["id"])]
    )
    class_freq_weight = class_freq.float() ** freq_weight_power
    return class_freq_weight


def get_fed_loss_cls_weights_v2(dataset_names: Union[str, List[str]], freq_weight_power=1.0):
    """
    Get frequency weight for each class sorted by class id.
    We now calcualte freqency weight using image_count to the power freq_weight_power.

    Args:
        dataset_names: list of dataset names
        freq_weight_power: power value
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    logger = logging.getLogger(__name__)

    class_freq_weight_list = []
    for dataset_name in dataset_names:
        if MetadataCatalog.get(dataset_name).get("json_file") is None:
            continue
        class_freq_path = MetadataCatalog.get(dataset_name).json_file[:-5] + "_cat_info.json"
        if os.path.exists(class_freq_path):
            logger.info(
                "Search outside metadata 'image_count' for dataset '{}' from '{}'".format(
                    dataset_name, class_freq_path
                )
            )
            # return load_fed_loss_cls_weights(class_freq_path, freq_weight_power)
            class_freq_weight_list.append(
                load_fed_loss_cls_weights(class_freq_path, freq_weight_power)
            )
            continue
        else:
            logger.info(
                "Nofind outside metadata 'image_count' for dataset '{}' from '{}'".format(
                    dataset_name, class_freq_path
                )
            )

        logger.info("Using builtin metadata 'image_count' for dataset '{}'".format(dataset_name))

        # check_metadata_consistency("class_image_count", dataset_names)

        meta = MetadataCatalog.get(dataset_name)
        class_freq_meta = meta.class_image_count
        class_freq = torch.tensor(
            [c["image_count"] for c in sorted(class_freq_meta, key=lambda x: x["id"])]
        )
        class_freq_weight = class_freq.float() ** freq_weight_power
        # return class_freq_weight
        class_freq_weight_list.append(class_freq_weight)

    return class_freq_weight_list[0] if len(class_freq_weight_list) == 1 else class_freq_weight_list


def build_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    assert not (cfg.INPUT.AUTOAUGMENT.ENABLED and cfg.INPUT.LSJ.ENABLED)

    augmentation = []
    if is_train and cfg.INPUT.AUTOAUGMENT.ENABLED:
        augmentation.append(AutoAugment(cfg))

        if cfg.INPUT.RANDOM_FLIP != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )
        if cfg.INPUT.RANDOM_COLOR.ENABLED:
            augmentation.append(T.RandomBrightness(0.5, 1.5))
            augmentation.append(T.RandomContrast(0.5, 1.5))
            augmentation.append(T.RandomSaturation(0.0, 2.0))
        return augmentation

    if is_train and cfg.INPUT.LSJ.ENABLED:
        augmentation.append(LargeScaleJitter(cfg))

        if cfg.INPUT.RANDOM_FLIP != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )
        if cfg.INPUT.RANDOM_COLOR.ENABLED:
            augmentation.append(T.RandomBrightness(0.5, 1.5))
            augmentation.append(T.RandomContrast(0.5, 1.5))
            augmentation.append(T.RandomSaturation(0.0, 2.0))
        return augmentation

    return build_augmentation_d2(cfg, is_train)


def build_augmentation_lsj(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    augmentation = []
    if is_train:
        augmentation.append(LargeScaleJitter(cfg))

        if cfg.INPUT.RANDOM_FLIP != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )
        if cfg.INPUT.RANDOM_COLOR.ENABLED:
            augmentation.append(T.RandomBrightness(0.5, 1.5))
            augmentation.append(T.RandomContrast(0.5, 1.5))
            augmentation.append(T.RandomSaturation(0.0, 2.0))
        return augmentation

    return build_augmentation_d2(cfg, is_train)


def build_augmentation_aa(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    augmentation = []
    if is_train:
        augmentation.append(AutoAugment(cfg))

        if cfg.INPUT.RANDOM_FLIP != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )
        if cfg.INPUT.RANDOM_COLOR.ENABLED:
            augmentation.append(T.RandomBrightness(0.5, 1.5))
            augmentation.append(T.RandomContrast(0.5, 1.5))
            augmentation.append(T.RandomSaturation(0.0, 2.0))
        return augmentation

    return build_augmentation_d2(cfg, is_train)


build_transform_gen = build_augmentation
"""
Alias for backward-compatibility.
"""
