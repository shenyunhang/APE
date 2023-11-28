import torch.nn as nn

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detrex.modeling.neck import ChannelMapper
from ape.layers import VisionLanguageFusion
from ape.modeling.ape_deta import (
    DeformableDETRSegmVL,
    DeformableDetrTransformerDecoderVL,
    DeformableDetrTransformerEncoderVL,
    DeformableDetrTransformerVL,
)
from ape.modeling.text import EVA02CLIP

from ...common.backbone.vitl_eva02_clip import backbone
from .ape_deta_vitl_eva02_lsj1024 import dataloader, lr_multiplier, model, optimizer, train

model.model_vision.backbone = backbone

train.init_checkpoint = (
    "models/QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14to16_s6B.pt?matching_heuristics=True"
)

model.model_language = L(EVA02CLIP)(
    clip_model="EVA02-CLIP-bigE-14-plus",
    cache_dir="models/QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt",
    dtype="float16",
)
model.model_vision.embed_dim_language = 1024

model.model_vision.neck = L(ChannelMapper)(
    input_shapes={
        "p2": ShapeSpec(channels=256),
        "p3": ShapeSpec(channels=256),
        "p4": ShapeSpec(channels=256),
        "p5": ShapeSpec(channels=256),
        "p6": ShapeSpec(channels=256),
    },
    in_features=["p2", "p3", "p4", "p5", "p6"],
    out_channels=256,
    num_outs=5,
    kernel_size=1,
    norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
)

model.model_vision.mask_in_features = ["p2"]
model.model_vision.input_shapes = {
    "p2": ShapeSpec(channels=256),
    "p3": ShapeSpec(channels=256),
    "p4": ShapeSpec(channels=256),
    "p5": ShapeSpec(channels=256),
    "p6": ShapeSpec(channels=256),
}

model.model_vision.transformer.encoder.num_layers = 6
model.model_vision.transformer.decoder.num_layers = 6
model.model_vision.transformer.encoder.embed_dim = 256
model.model_vision.transformer.decoder.embed_dim = 256
model.model_vision.embed_dim = 256
model.model_vision.backbone.out_channels = 256

model.model_vision.update(
    _target_=DeformableDETRSegmVL,
)
model.model_vision.transformer.update(
    _target_=DeformableDetrTransformerVL,
)
model.model_vision.transformer.encoder.update(
    _target_=DeformableDetrTransformerEncoderVL,
)
model.model_vision.transformer.decoder.update(
    _target_=DeformableDetrTransformerDecoderVL,
)

model.model_vision.transformer.encoder.vl_layer = L(VisionLanguageFusion)(
    v_dim="${....embed_dim}",
    l_dim="${....embed_dim_language}",
    embed_dim=2048,
    num_heads=8,
    dropout=0.1,
    drop_path=0.0,
    init_values=1.0 / 6,
    stable_softmax_2d=True,
    clamp_min_for_underflow=True,
    clamp_max_for_overflow=True,
    use_checkpoint=True,
)

model.model_vision.text_feature_bank = True
model.model_vision.text_feature_bank_reset = True
model.model_vision.text_feature_reduce_before_fusion = True
model.model_vision.text_feature_batch_repeat = True
model.model_vision.expression_cumulative_gt_class = True
model.model_vision.name_prompt_fusion_type = "zero"

model.model_vision.stuff_dataset_learn_thing = False
model.model_vision.stuff_prob_thing = -1.0
model.model_vision.transformer.proposal_ambiguous = 2

train.output_dir = "output/" + __file__[:-3]
model.model_vision.vis_period = 12800
