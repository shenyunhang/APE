from detectron2.config import LazyCall as L
from ape.modeling.text import EVA01CLIP

from ...COCO_InstanceSegmentation.ape_deta.ape_deta_vitl_eva02_lsj1024_cp_12ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)
from ...common.data.coco_semantic_lsj1024 import dataloader

num_classes = 54
model.model_vision.num_classes = num_classes
model.model_vision.criterion[0].num_classes = num_classes
model.model_vision.criterion[0].matcher_stage2.num_classes = num_classes

model.model_vision.instance_on = False
model.model_vision.semantic_on = True
model.model_vision.panoptic_on = False

train.output_dir = "output/" + __file__[:-3]

model.model_vision.dataset_prompts = ["name"]
model.model_vision.dataset_names = ["stuffonly"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

train.output_dir = "output/" + __file__[:-3]
model.model_vision.output_dir = train.output_dir

model.model_language = L(EVA01CLIP)(
    clip_model="EVA_CLIP_g_14_X", cache_dir="models/BAAI/EVA/eva_clip_psz14.pt"
)
model.model_vision.embed_dim_language = 1024
