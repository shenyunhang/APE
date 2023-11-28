import copy
import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss

import fvcore.nn.weight_init as weight_init
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.layers import Conv2d, ShapeSpec, get_norm, move_device_like
from detectron2.modeling import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.structures import BitMasks, Boxes, ImageList, Instances
from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid
from torchvision.ops.boxes import batched_nms
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput


class T5_warpper(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        bg_word="",
        dtype="bfloat16",
        loss_type="CE",
        use_fed_loss=False,
        fed_loss_num_classes=1000,
        inference_text=False,
        inference_prob=False,
        inference_prob_fast=False,
        train_positive_only=False,
        test_constraint=False,
        vision_port="encoder",
        eval_only=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dtype = getattr(torch, dtype)

        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )

        if eval_only:
            self.t5_model.eval()
        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.to(self.dtype)

        self.eos_token_id = self.tokenizer("\n", add_special_tokens=False).input_ids[0]

        self.text_list_to_feature = {}

    @autocast(enabled=False)
    @torch.no_grad()
    def forward_text(self, text_list, cache=False):
        if cache and tuple(text_list) in self.text_list_to_feature:
            return self.text_list_to_feature[tuple(text_list)]

        text_token = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding="longest",
        ).to(self.device)
        input_ids = text_token.input_ids
        attention_mask = text_token.attention_mask

        encoder_outputs = self.t5_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=None,
            head_mask=None,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_state = encoder_outputs.last_hidden_state

        feature = agg_lang_feat(last_hidden_state, attention_mask).clone().detach()

        if cache:
            self.text_list_to_feature[tuple(text_list)] = feature

        return feature

    @property
    def device(self):
        return self.t5_model.device
