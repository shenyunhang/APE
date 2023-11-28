from functools import partial

import torch.nn as nn

from detectron2.config import LazyCall as L
from detectron2.data.catalog import MetadataCatalog
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detrex.config import get_config
from ape.modeling.backbone.vit import get_vit_lr_decay_rate
from ape.modeling.backbone.vit_eva02 import SimpleFeaturePyramid, ViT

from .....detectron2.configs.common.data.constants import constants
from ...common.backbone.vitl_eva02 import backbone
from ...common.data.coco_instance_lsj1024_cp import dataloader
from .models.deformable_deta_segm_r50 import model

model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"

model.backbone = backbone

model.neck = None

model.mask_in_features = ["p2"]
model.input_shapes = {
    "p2": ShapeSpec(channels=256),
    "p3": ShapeSpec(channels=256),
    "p4": ShapeSpec(channels=256),
    "p5": ShapeSpec(channels=256),
    "p6": ShapeSpec(channels=256),
}

optimizer = get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = (
    lambda module_name: 0.1
    if "reference_points" in module_name or "sampling_offsets" in module_name
    else get_vit_lr_decay_rate(module_name, lr_decay_rate=0.8, num_layers=24)
    if "backbone.net" in module_name
    else 1
)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
optimizer.params.weight_decay_norm = None

optimizer.lr = 2e-4
optimizer.weight_decay = 1e-4

train = get_config("common/train.py").train
train.max_iter = 90000
train.eval_period = 5000
train.log_period = 20

train.checkpointer.period = 5000
train.checkpointer.max_to_keep = 2

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.device = "cuda"

train.init_checkpoint = (
    "models/Yunxin-CV/EVA-02/eva02/pt/eva02_L_pt_in21k_p14to16.pt?matching_heuristics=True"
)

train.amp.enabled = True
train.ddp.fp16_compression = True

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
lr_multiplier.scheduler.milestones = [75000, 90000]
lr_multiplier.warmup_length = 1000 / train.max_iter

dataloader.train.num_workers = 16
dataloader.train.total_batch_size = 16
dataloader.train.mapper.image_format = "RGB"
dataloader.train.mapper.use_instance_mask = True

if isinstance(dataloader.train.dataset.names, str):
    model.metadata = MetadataCatalog.get(dataloader.train.dataset.names)
else:
    model.metadata = MetadataCatalog.get(dataloader.train.dataset.names[0])

train.output_dir = "output/" + __file__[:-3]
dataloader.train.mapper.output_dir = train.output_dir
