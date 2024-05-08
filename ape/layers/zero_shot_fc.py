import logging
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class ZeroShotFC(nn.Module):
    def __init__(
        self,
        input_size,
        *,
        num_classes: int,
        zs_weight_path: str,
        zs_weight_dim: int = 512,
        use_bias: float = 0.0,
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
        use_project: bool = True,
        use_sigmoid_ce: bool,
        prior_prob: float = 0.01,
        zs_vocabulary: str = "",
        text_model: str = "",
    ):
        super().__init__()

        # assert use_sigmoid_ce
        # assert cls_agnostic_bbox_reg

        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature
        self.use_project = use_project
        self.zs_weight_dim = zs_weight_dim

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias, requires_grad=True)

        if self.use_project:
            self.linear = nn.Linear(input_size, zs_weight_dim)

            if use_sigmoid_ce:
                bias_value = -math.log((1 - prior_prob) / prior_prob)
            else:
                bias_value = 0
            torch.nn.init.constant_(self.linear.bias, bias_value)
            torch.nn.init.normal_(self.linear.weight, std=0.01)

        if len(zs_vocabulary) > 0:
            from ape.modeling.text import get_clip_embeddings

            logger.info("Generating weight for " + zs_vocabulary)
            zs_vocabulary = zs_vocabulary.split(",")
            num_classes = len(zs_vocabulary)
            zs_weight = get_clip_embeddings(text_model, zs_vocabulary)
            zs_weight = zs_weight.permute(1, 0).contiguous()
        elif zs_weight_path == "rand":
            zs_weight = torch.randn((zs_weight_dim, num_classes))
            nn.init.normal_(zs_weight, std=0.01)
        elif zs_weight_path == "zeros":
            zs_weight = torch.zeros((zs_weight_dim, num_classes))
        elif zs_weight_path == "online":
            from ape.modeling.text import build_clip_text_encoder

            zs_weight = torch.zeros((zs_weight_dim, num_classes))
            self.text_encoder = build_clip_text_encoder(text_model, pretrain=True)
            self.text_encoder.eval()
        else:
            logger.info("Loading " + zs_weight_path)
            zs_weight = (
                torch.tensor(np.load(zs_weight_path), dtype=torch.float32)
                .permute(1, 0)
                .contiguous()
            )
            logger.info(f"Loaded zs_weight {zs_weight.size()}")

        zs_weight = torch.cat([zs_weight, zs_weight.new_zeros((self.zs_weight_dim, 1))], dim=1)
        logger.info(f"Cated zs_weight {zs_weight.size()}")

        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)

        if zs_weight_path == "rand":
            self.zs_weight = nn.Parameter(zs_weight, requires_grad=True)
        else:
            self.register_buffer("zs_weight", zs_weight)

        assert (
            self.zs_weight.shape[1] == num_classes + 1
        ), f"zs_weight={self.zs_weight.shape} v.s. num_classes={num_classes}"

    def forward(self, x, classifier=None):
        """
        Inputs:
            x: B x D or B x N x D
            classifier: C x D
        """
        x_shape = x.shape
        if len(x_shape) == 3:
            x = x.reshape(x_shape[0] * x_shape[1], x_shape[2])
        assert x.dim() == 2

        if self.use_project:
            x = self.linear(x)
        if classifier is not None:
            if isinstance(classifier, str):
                from ape.modeling.text import get_clip_embeddings

                zs_weight = get_clip_embeddings(
                    self.text_encoder, classifier, prompt="", device=x.device
                )
            else:
                zs_weight = classifier
            zs_weight = zs_weight.permute(1, 0).contiguous()
            zs_weight = torch.cat([zs_weight, zs_weight.new_zeros((self.zs_weight_dim, 1))], dim=1)
            if self.norm_weight:
                zs_weight = F.normalize(zs_weight, p=2, dim=0)
        else:
            zs_weight = self.zs_weight
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = torch.mm(x, zs_weight)
        if self.use_bias:
            x = x + self.cls_bias

        if len(x_shape) == 3:
            x = x.reshape(x_shape[:2] + zs_weight.shape[1:])
        return x

    def set_predictor(self, param_or_path):
        if type(param_or_path) == str:
            logger.info("Loading " + param_or_path)
            zs_weight = (
                torch.tensor(np.load(param_or_path), dtype=torch.float32).permute(1, 0).contiguous()
            )
        else:
            zs_weight = param_or_path.permute(1, 0).contiguous()
        logger.info(f"Loaded zs_weight {zs_weight.size()}")

        zs_weight = torch.cat([zs_weight, zs_weight.new_zeros((self.zs_weight_dim, 1))], dim=1)
        logger.info(f"Cated zs_weight {zs_weight.size()}")

        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)

        zs_weight = zs_weight.to(self.zs_weight.device)
        self.zs_weight = zs_weight

    def extra_repr(self):
        extra_repr = ""
        valtype = (int, float, bool, str, dict, list)
        for attribute, value in self.__dict__.items():
            if type(value) in valtype:
                extra_repr += "{}={}, ".format(attribute, value)
        return extra_repr[:-2]
