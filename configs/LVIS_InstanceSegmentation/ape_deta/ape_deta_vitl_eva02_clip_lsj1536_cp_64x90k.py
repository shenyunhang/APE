from detectron2.data.detection_utils import get_fed_loss_cls_weights

from ...COCO_InstanceSegmentation.ape_deta.ape_deta_vitl_eva02_clip_lsj1536_cp_64x90k import (
    lr_multiplier,
    model,
    optimizer,
    train,
)
from ...common.data.lvis_instance_lsj1536_cp import dataloader

model.model_vision.num_classes = 1203
model.model_vision.select_box_nums_for_evaluation = 300
model.model_vision.criterion[0].num_classes = 1203
model.model_vision.criterion[0].use_fed_loss = True
model.model_vision.criterion[0].get_fed_loss_cls_weights = lambda: get_fed_loss_cls_weights(
    dataloader.train.dataset.names, 0.5
)
model.model_vision.criterion[0].fed_loss_num_classes = 50

del optimizer.params.weight_decay_norm

optimizer.weight_decay = 0.05

train.max_iter = 90000
train.eval_period = 10000

lr_multiplier.scheduler.milestones = [75000, 90000]
lr_multiplier.warmup_length = 1000 / train.max_iter

model.model_vision.dataset_prompts = ["name"]
model.model_vision.dataset_names = ["lvis"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

dataloader.train.total_batch_size = 64

train.output_dir = "output/" + __file__[:-3]
