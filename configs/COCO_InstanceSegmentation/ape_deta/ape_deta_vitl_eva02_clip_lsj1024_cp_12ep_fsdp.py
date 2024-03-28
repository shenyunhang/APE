from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec

from detectron2.model_zoo import get_config as get_config_d2
from detrex.config import get_config as get_config_detrex
from ape.modeling.backbone.vit import get_vit_lr_decay_rate

from ape.modeling.text import EVA02CLIP

from ...common.backbone.vitl_eva02_clip import backbone
from ...common.data.coco_instance_lsj1024_cp import dataloader
from .models.ape_deta_r50 import model

constants = get_config_d2("common/data/constants.py").constants

model.model_vision.pixel_mean = constants.imagenet_rgb256_mean
model.model_vision.pixel_std = constants.imagenet_rgb256_std
model.model_vision.input_format = "RGB"

model.model_vision.backbone = backbone

model.model_vision.neck.input_shapes = {
    "p2": ShapeSpec(channels=256),
    "p3": ShapeSpec(channels=256),
    "p4": ShapeSpec(channels=256),
    "p5": ShapeSpec(channels=256),
    "p6": ShapeSpec(channels=256),
}
model.model_vision.neck.in_features = ["p2", "p3", "p4", "p5", "p6"]

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
optimizer.betas = (0.9, 0.999)
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
    "models/QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14to16_s6B.pt?matching_heuristics=True"
)

train.amp.enabled = True
train.ddp.fp16_compression = True
train.fsdp = dict(
    cpu_offload=False,
    use_orig_params=True,
    sync_module_states=True,
    module_name_to_wrap=["Block",],
    # module_name_to_wrap=["Block", "BaseTransformerLayer"],
    param_dtype="float32",
    reduce_dtype="float32",
    buffer_dtype="float32",
    # param_dtype="float16",
    # reduce_dtype="float16",
    # buffer_dtype="float16",
)

lr_multiplier = get_config_detrex("common/coco_schedule.py").lr_multiplier_12ep
lr_multiplier.scheduler.milestones = [75000, 90000]
lr_multiplier.warmup_length = 1000 / train.max_iter

dataloader.train.num_workers = 16
dataloader.train.total_batch_size = 16
dataloader.train.mapper.image_format = "RGB"
dataloader.train.mapper.use_instance_mask = True

model.model_vision.dataset_prompts = ["name"]
model.model_vision.dataset_names = ["coco_2017"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

train.output_dir = "output/" + __file__[:-3]

model.model_language = L(EVA02CLIP)(
    clip_model="EVA02-CLIP-bigE-14-plus",
    cache_dir="models/QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt",
    # dtype="float16",
)
model.model_vision.embed_dim_language = 1024

from ape.data.build import build_detection_test_loader
dataloader.test.update(
    _target_=build_detection_test_loader,
)
