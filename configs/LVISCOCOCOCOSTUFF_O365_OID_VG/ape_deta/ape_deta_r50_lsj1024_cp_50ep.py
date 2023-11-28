from detectron2.config import LazyCall as L
from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detrex.config import get_config
from ape.modeling.text import EVA01CLIP

from ...common.data.lviscocococostuff_o365_oid_vg_panoptic_lsj1024_cp import dataloader
from .models.ape_deta_r50 import model

model.model_vision.num_classes = 1256
model.model_vision.criterion[0].num_classes = 1256
model.model_vision.select_box_nums_for_evaluation = 300
model.model_vision.criterion[0].use_fed_loss = True
model.model_vision.criterion[0].get_fed_loss_cls_weights = lambda: get_fed_loss_cls_weights(
    dataloader.train.dataset.names[0], 0.5
)
model.model_vision.criterion[0].fed_loss_num_classes = 50

optimizer = get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = (
    lambda module_name: 0.1
    if "backbone" in module_name
    or "reference_points" in module_name
    or "sampling_offsets" in module_name
    else 1
)
optimizer.params.weight_decay_norm = None

optimizer.lr = 2e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4

train = get_config("common/train.py").train
train.max_iter = 375000
train.eval_period = 2000000
train.log_period = 20

train.checkpointer.period = 5000
train.checkpointer.max_to_keep = 2

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.device = "cuda"

train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.init_checkpoint = "models/torchvision/R-50.pkl"

train.amp.enabled = True
train.ddp.fp16_compression = True

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep

dataloader.train.total_batch_size = 16
dataloader.train.total_batch_size_list = [16, 16, 16, 16]
dataloader.train.mapper.image_format = "RGB"
dataloader.train.mapper.use_instance_mask = True

model.model_vision.dataset_prompts = ["name", "name", "name", "name"]
model.model_vision.dataset_names = [
    "lvis+stuffonly",
    "objects365",
    "openimages",
    "visualgenome",
]
model.model_vision.dataset_metas = dataloader.train.dataset.names

train.output_dir = "output/" + __file__[:-3]
model.model_vision.output_dir = train.output_dir
dataloader.train.mapper.output_dir = train.output_dir
dataloader.train.mapper.vis_period = 12800

model.model_language = L(EVA01CLIP)(
    clip_model="EVA_CLIP_g_14_X", cache_dir="models/BAAI/EVA/eva_clip_psz14.pt"
)
model.model_vision.embed_dim_language = 1024
