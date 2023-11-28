from detectron2.config import LazyCall as L
from detrex.config import get_config
from ape.modeling.text import EVA01CLIP

from ...common.data.coco_instance import dataloader
from .models.ape_deta_r50 import model

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
lr_multiplier.scheduler.milestones = [75000, 90000]
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.init_checkpoint = "models/torchvision/R-50.pkl"

train.max_iter = 90000

train.eval_period = 5000

train.log_period = 20

train.checkpointer.period = 5000
train.checkpointer.max_to_keep = 2

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.device = "cuda"

optimizer.lr = 2e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = (
    lambda module_name: 0.1
    if "backbone" in module_name
    or "reference_points" in module_name
    or "sampling_offsets" in module_name
    else 1
)
optimizer.params.weight_decay_norm = None

dataloader.train.num_workers = 16

dataloader.train.total_batch_size = 16

dataloader.train.mapper.use_instance_mask = True

train.amp.enabled = True
train.ddp.fp16_compression = True

model.model_vision.dataset_prompts = ["name"]
model.model_vision.dataset_names = ["coco_2017"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

train.output_dir = "output/" + __file__[:-3]

model.model_language = L(EVA01CLIP)(
    clip_model="EVA_CLIP_g_14_X", cache_dir="models/BAAI/EVA/eva_clip_psz14.pt"
)
model.model_vision.embed_dim_language = 1024
model.model_vision.text_feature_reduce_type = "last"
model.model_vision.text_feature_reduce_before_fusion = True
