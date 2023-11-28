from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.model_zoo import get_config as get_config_d2
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detrex.config import get_config as get_config_detrex
from ape.modeling.backbone.vit import get_vit_lr_decay_rate
from ape.modeling.backbone.vit_eva02 import SimpleFeaturePyramid, ViT
from ape.modeling.text import EVA01CLIP

from ...common.backbone.vitl_eva02 import backbone

from ...common.data.refcoco_group_by_image_instance_lsj1024 import dataloader
from .ape_deta_r50_vlf_12ep import model

model.model_vision.num_classes = 256
model.model_vision.select_box_nums_for_evaluation = 1
model.model_vision.criterion[0].num_classes = 256
model.model_vision.criterion[1].num_classes = 256

constants = get_config_d2("common/data/constants.py").constants

model.model_vision.pixel_mean = constants.imagenet_rgb256_mean
model.model_vision.pixel_std = constants.imagenet_rgb256_std
model.model_vision.input_format = "RGB"

model.model_vision.backbone = backbone

model.model_vision.neck = None

model.model_vision.mask_in_features = ["p2"]
model.model_vision.input_shapes = {
    "p2": ShapeSpec(channels=256),
    "p3": ShapeSpec(channels=256),
    "p4": ShapeSpec(channels=256),
    "p5": ShapeSpec(channels=256),
    "p6": ShapeSpec(channels=256),
}

optimizer = get_config_detrex("common/optim.py").AdamW
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

train = get_config_detrex("common/train.py").train
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

lr_multiplier = get_config_detrex("common/coco_schedule.py").lr_multiplier_12ep
lr_multiplier.scheduler.milestones = [75000, 90000]
lr_multiplier.warmup_length = 1000 / train.max_iter

dataloader.train.num_workers = 16
dataloader.train.total_batch_size = 16
dataloader.train.total_batch_size_list = ["${..total_batch_size}", "${..total_batch_size}"]
dataloader.train.mapper.image_format = "RGB"
dataloader.train.mapper.use_instance_mask = True

dataloader.train.mapper.max_num_phrase = 128
dataloader.train.mapper.nms_thresh_phrase = 0.6

model.model_vision.dataset_prompts = ["phrase", "expression"]
model.model_vision.dataset_names = ["refcoco-mixed_group-by-image", "refcoco"]
model.model_vision.dataset_metas = dataloader.train.dataset.names + ["refcoco-mixed"]

model.model_language = L(EVA01CLIP)(
    clip_model="EVA_CLIP_g_14_X",
    cache_dir="models/BAAI/EVA/eva_clip_psz14.pt",
)
model.model_vision.embed_dim_language = 1024

model.model_vision.text_feature_bank = True
model.model_vision.text_feature_reduce_before_fusion = True
model.model_vision.text_feature_batch_repeat = True
model.model_vision.expression_cumulative_gt_class = True

model.model_vision.transformer.encoder.use_act_checkpoint = True
model.model_vision.transformer.decoder.use_act_checkpoint = True

train.output_dir = "output/" + __file__[:-3]
