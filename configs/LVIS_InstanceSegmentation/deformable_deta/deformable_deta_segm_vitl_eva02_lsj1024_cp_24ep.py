from detectron2.data.detection_utils import get_fed_loss_cls_weights

from ...COCO_InstanceSegmentation.deformable_deta.deformable_deta_segm_vitl_eva02_lsj1024_cp_12ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)
from ...common.data.lvis_instance_lsj1024_cp import dataloader

model.num_classes = 1203
model.select_box_nums_for_evaluation = 300
model.criterion.num_classes = 1203
model.criterion.use_fed_loss = True
model.criterion.get_fed_loss_cls_weights = lambda: get_fed_loss_cls_weights(
    dataloader.train.dataset.names, 0.5
)
model.criterion.fed_loss_num_classes = 50

del optimizer.params.weight_decay_norm

optimizer.weight_decay = 0.05

train.max_iter = 180000
train.eval_period = 20000

lr_multiplier.scheduler.milestones = [150000, 180000]
lr_multiplier.warmup_length = 1000 / train.max_iter

model.dataset_metas = dataloader.train.dataset.names

train.output_dir = "output/" + __file__[:-3]
dataloader.train.mapper.vis_period = 12800
