from functools import partial

import torch.nn as nn

from detectron2.config import LazyCall as L
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from ape.modeling.backbone.vit_eva_clip import SimpleFeaturePyramid, ViT

backbone = L(SimpleFeaturePyramid)(
    net=L(ViT)(  # Single-scale ViT backbone
        img_size=1536,
        patch_size=16,
        embed_dim=1792,
        depth=64,
        num_heads=16,
        drop_path_rate=0.4,
        window_size=32,
        mlp_ratio=8.571428571428571,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=list(range(0, 3))
        + list(range(4, 7))
        + list(range(8, 11))
        + list(range(12, 15))
        + list(range(16, 19))
        + list(range(20, 23))
        + list(range(24, 27))
        + list(range(28, 31))
        + list(range(32, 35))
        + list(range(36, 39))
        + list(range(40, 43))
        + list(range(44, 47))
        + list(range(48, 51))
        + list(range(52, 55))
        + list(range(56, 59))
        + list(range(60, 63)),
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
        use_act_checkpoint=True,
        xattn=True,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        postnorm=True,
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=1536,
)
