from functools import partial

import torch.nn as nn

from detectron2.config import LazyCall as L
from detectron2.modeling import SimpleFeaturePyramid, ViT
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from .....detectron2.configs.common.data.constants import constants
from .....detectron2.projects.ViTDet.configs.common.coco_loader_lsj import dataloader
from .deformable_deta_r50_12ep import lr_multiplier, model, optimizer, train

model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"
dataloader.train.mapper.image_format = "RGB"
dataloader.train.total_batch_size = 16


embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
model.backbone = L(SimpleFeaturePyramid)(
    net=L(ViT)(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=1024,
)

model.neck = None

optimizer.params.lr_factor_func = (
    lambda module_name: 0.1
    if "reference_points" in module_name or "sampling_offsets" in module_name
    else get_vit_lr_decay_rate(module_name, lr_decay_rate=0.7, num_layers=12)
    if "backbone" in module_name
    else 1
)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}


lr_multiplier.warmup_length = 1000 / train.max_iter

train.amp.enabled = False
train.ddp.fp16_compression = False

train.init_checkpoint = (
    "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
)
train.init_checkpoint = "models/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"

train.output_dir = "output/" + __file__[:-3]
dataloader.evaluator.output_dir = train.output_dir
