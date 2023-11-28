import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, ShapeSpec, get_norm, move_device_like
from detectron2.modeling import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.structures import BitMasks, Boxes, ImageList, Instances
from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid
from torchvision.ops.boxes import batched_nms


class SomeThing(nn.Module):
    def __init__(
        self,
        model_vision,
        model_language,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_vision = model_vision
        self.model_language = model_language

        self.model_vision.set_model_language(self.model_language)
        del self.model_language

    def forward(self, batched_inputs, do_postprocess=True):
        losses = self.model_vision(batched_inputs, do_postprocess=do_postprocess)
        return losses

    def set_eval_dataset(self, dataset_name):
        self.model_vision.set_eval_dataset(dataset_name)
