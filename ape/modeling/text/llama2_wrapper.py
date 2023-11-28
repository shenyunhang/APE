import copy
import logging
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
from detectron2.utils import comm
from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid
from torchvision.ops.boxes import batched_nms
from transformers import BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.modeling_outputs import BaseModelOutput


class Llama2(nn.Module):
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
        load_in_4bit=False,
        load_in_8bit=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dtype = getattr(torch, dtype)

        self.config = LlamaConfig.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )

        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
            )
            device_map = {"": comm.get_local_rank()}
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf4",
                bnb_8bit_compute_dtype=self.dtype,
                bnb_8bit_use_double_quant=True,
            )
            device_map = {"": comm.get_local_rank()}
        else:
            quantization_config = None
            device_map = None
        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            quantization_config=quantization_config,
            device_map=device_map,
        )

        if quantization_config is None:
            for name, param in self.model.named_parameters():
                param.data = param.data.to(self.dtype)

        self.tokenizer = LlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )

        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.tokenizer.padding_side = "left"

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if eval_only:
            self.model.eval()
            for name, param in self.model.named_parameters():
                param.requires_grad = False

        logger = logging.getLogger(__name__)
        logger.info("memory footprint: {}G".format(self.model.get_memory_footprint() / 1024**3))

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

        max_batch_size = 128
        if torch.cuda.mem_get_info(self.device)[0] / 1024**3 < 5:
            max_batch_size = 128

        chunck_num = input_ids.size(0) // max_batch_size + 1
        last_hidden_state = []
        for chunck_id in range(chunck_num):
            outputs = self.model(
                input_ids=input_ids[chunck_id * max_batch_size : (chunck_id + 1) * max_batch_size],
                attention_mask=attention_mask[
                    chunck_id * max_batch_size : (chunck_id + 1) * max_batch_size
                ],
                inputs_embeds=None,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden_state.append(outputs.hidden_states[-1].clone().detach())

        last_hidden_state = torch.cat(last_hidden_state, dim=0)

        last_hidden_state = torch.nan_to_num(last_hidden_state, nan=0.0, posinf=0.0, neginf=0.0)

        ret = {
            "attention_mask": attention_mask,
            "last_hidden_state": last_hidden_state,
        }

        if cache:
            self.text_list_to_feature[tuple(text_list)] = ret

        return ret

    @property
    def device(self):
        return self.model.device
