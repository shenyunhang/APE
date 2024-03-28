from functools import partial

import torch.nn as nn

from detectron2.config import LazyCall as L
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from ape.modeling.backbone.vit_eva02 import SimpleFeaturePyramid, ViT

# Creates Simple Feature Pyramid from ViT backbone
backbone = L(SimpleFeaturePyramid)(
    net=L(ViT)(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        drop_path_rate=0.8,
        window_size=14,
        mlp_ratio=4 * 2 / 3,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=list(range(0, 2))
        + list(range(3, 5))
        + list(range(6, 8))
        + list(range(9, 11)),
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
        use_act_checkpoint=False,
        xattn=True,
        subln=False,
        swiglu=True,
        naiveswiglu=False,
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=1024,
)
