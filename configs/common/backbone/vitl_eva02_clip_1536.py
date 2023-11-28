from functools import partial

import torch.nn as nn

from detectron2.config import LazyCall as L
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from ape.modeling.backbone.vit_eva_clip import SimpleFeaturePyramid, ViT

backbone = L(SimpleFeaturePyramid)(
    net=L(ViT)(  # Single-scale ViT backbone
        img_size=1536,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        drop_path_rate=0.4,
        window_size=32,
        mlp_ratio=4 * 2 / 3,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=list(range(0, 2))
        + list(range(3, 5))
        + list(range(6, 8))
        + list(range(9, 11))
        + list(range(12, 14))
        + list(range(15, 17))
        + list(range(18, 20))
        + list(range(21, 23)),
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
        use_act_checkpoint=True,
        xattn=True,
        rope=True,
        pt_hw_seq_len=16,
        intp_freq=True,
        naiveswiglu=True,
        subln=True,
        pretrain_img_size=336,
        pretrain_use_cls_token=True,
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=1536,
)
