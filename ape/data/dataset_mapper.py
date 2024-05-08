# Copyright (c) Facebook, Inc. and its affiliates.
import logging

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper as DatasetMapper_d2

from . import detection_utils as utils_ape

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper_ape"]


class DatasetMapper_ape(DatasetMapper_d2):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train: bool = True):
        super().__init__(cfg, is_train)
        augmentations = utils_ape.build_augmentation(cfg, is_train)
        self.augmentations = T.AugmentationList(augmentations)

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
