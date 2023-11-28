from functools import partial

from ape.modeling.backbone.vit_eva import SimpleFeaturePyramid, ViT, get_vit_lr_decay_rate

from ..common.coco_loader_lsj1280 import dataloader
from .deformable_deta_vitb_lsj1024_12ep import lr_multiplier, model, optimizer, train

model.backbone.update(
    _target_=SimpleFeaturePyramid,
)
model.backbone.net.update(
    _target_=ViT,
)

dataloader.train.total_batch_size = 16

model.backbone.net.beit_like_qkv_bias = True
model.backbone.net.beit_like_gamma = False
model.backbone.net.freeze_patch_embed = True
model.backbone.square_pad = 1280
model.backbone.net.img_size = 1280
model.backbone.net.patch_size = 16
model.backbone.net.window_size = 16
model.backbone.net.embed_dim = 1408
model.backbone.net.depth = 40
model.backbone.net.num_heads = 16
model.backbone.net.mlp_ratio = 6144 / 1408
model.backbone.net.use_act_checkpoint = True
model.backbone.net.drop_path_rate = 0.6  # 0.5 --> 0.6
model.backbone.net.window_block_indexes = (
    list(range(0, 3))
    + list(range(4, 7))
    + list(range(8, 11))
    + list(range(12, 15))
    + list(range(16, 19))
    + list(range(20, 23))
    + list(range(24, 27))
    + list(range(28, 31))
    + list(range(32, 35))
    + list(range(36, 39))
)

optimizer.lr = 2e-4
optimizer.params.lr_factor_func = (
    lambda module_name: 0.1
    if "reference_points" in module_name or "sampling_offsets" in module_name
    else get_vit_lr_decay_rate(module_name, lr_decay_rate=0.9, num_layers=40)
    if "backbone" in module_name
    else 1
)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
optimizer.params.weight_decay_norm = None

train.amp.enabled = False
train.ddp.fp16_compression = False

model.backbone.net.use_act_checkpoint = False
model.backbone.net.frozen_stages = 41

train.init_checkpoint = "models/BAAI/EVA/eva_o365.pth?matching_heuristics=True"
train.output_dir = "output/" + __file__[:-3]
dataloader.evaluator.output_dir = train.output_dir
