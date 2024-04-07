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
import functools

import torch
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy, ModuleWrapPolicy
except ImportError as e:
    print(e, "just skip this")

from ape.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate
from detectron2.utils import comm

from transformers.trainer_pt_utils import get_module_class_from_name

__all__ = [
    "create_fsdp_model",
    "DefaultPredictor",
]


def create_fsdp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa

    sharding_strategy_dict = {
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    }

    dtype_dict = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    auto_wrap_policy = None
    module_name_to_wrap = kwargs.pop("module_name_to_wrap", None)
    if module_name_to_wrap is not None:
        module_cls_to_wrap = set()
        for module_name in module_name_to_wrap:
            module_cls = get_module_class_from_name(model, module_name)
            if module_cls is None:
                raise Exception("Could not find the layer class to wrap in the model.")
            else:
                module_cls_to_wrap.add(module_cls)

        # print("module_cls_to_wrap", module_cls_to_wrap)
        # auto_wrap_policy = functools.partial(
        #     transformer_auto_wrap_policy,
        #     # Transformer layer class to wrap
        #     transformer_layer_cls=module_cls_to_wrap,
        # )
        auto_wrap_policy = ModuleWrapPolicy(module_cls_to_wrap)
    else:
        # auto_wrap_policy = functools.partial(
        #     size_based_auto_wrap_policy, min_num_params=int(1e5)
        # )
        auto_wrap_policy = size_based_auto_wrap_policy

    if comm.get_world_size() == 1:
        return model
    if "device_id" not in kwargs:
        kwargs["device_id"] = comm.get_local_rank()

    param_dtype = kwargs.pop("param_dtype", None)
    reduce_dtype = kwargs.pop("reduce_dtype", None)
    buffer_dtype = kwargs.pop("buffer_dtype", None)

    if param_dtype is not None:
        param_dtype = getattr(torch, param_dtype)
    if reduce_dtype is not None:
        reduce_dtype = getattr(torch, reduce_dtype)
    if buffer_dtype is not None:
        buffer_dtype = getattr(torch, buffer_dtype)

    # from ape.layers import MultiScaleDeformableAttention
    mp_policy = MixedPrecision(
        param_dtype=param_dtype,
        # Gradient communication precision.
        reduce_dtype=reduce_dtype,
        # Buffer precision.
        buffer_dtype=buffer_dtype,
        cast_forward_inputs=True,
         # _module_classes_to_ignore=(MultiScaleDeformableAttention,),
    )

    model = model.to(param_dtype)

    fsdp = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        **kwargs,
    )
    return fsdp

    model.model_vision.model_language = FSDP(
        model.model_vision.model_language,
        # auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        mixed_precision=mp_policy,
        **kwargs,
    )
    model.model_vision.backbone = FSDP(
        model.model_vision.backbone,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        **kwargs,
    )
    model.model_vision.transfomer = FSDP(
        model.model_vision.transformer,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        **kwargs,
    )

    # auto_wrap_policy = functools.partial(
    #     size_based_auto_wrap_policy, min_num_params=int(1e5)
    # )
    fsdp = FSDP(
        model,
        # auto_wrap_policy=size_based_auto_wrap_policy,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        mixed_precision=mp_policy,
        **kwargs,
    )

    # if fp16_compression:
    #     from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

    #     ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return fsdp


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
