# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import copy
import os
import sys

import torch

from ape.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate

__all__ = [
    "DefaultPredictor",
]


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)  # cfg can be modified by model
        self.model = instantiate(cfg.model)
        self.model.to(cfg.train.device)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.train.init_checkpoint)

        self.aug = instantiate(cfg.dataloader.test.mapper.augmentations[0])
        if "model_vision" in cfg.model:
            self.input_format = cfg.model.model_vision.input_format
        else:
            self.input_format = cfg.model.input_format
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, text_prompt=None, mask_prompt=None):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            if text_prompt is not None:
                inputs["prompt"] = "text"
                inputs["text_prompt"] = text_prompt
            if mask_prompt is not None:
                mask_prompt = self.aug.get_transform(mask_prompt).apply_image(mask_prompt)
                inputs["mask_prompt"] = torch.as_tensor(mask_prompt.astype("float32"))
            predictions = self.model([inputs])[0]
            return predictions
