from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detrex.config import get_config

from ...common.data.lviscocococostuff_panoptic_lsj1024_cp import dataloader
from ...LVIS_InstanceSegmentation.ape_deta.ape_deta_r50_24ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)

model.model_vision.num_classes = 1256
model.model_vision.criterion[0].num_classes = 1256
model.model_vision.criterion[0].use_fed_loss = True
model.model_vision.criterion[0].get_fed_loss_cls_weights = lambda: get_fed_loss_cls_weights(
    dataloader.train.dataset.names, 0.5
)
model.model_vision.criterion[0].fed_loss_num_classes = 50
model.model_vision.criterion[0].fed_loss_pad_type = "cat"

model.model_vision.instance_on = True
model.model_vision.semantic_on = True
model.model_vision.panoptic_on = False

train.max_iter = 375000
train.eval_period = 20000

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep

dataloader.train.total_batch_size = 16

model.model_vision.dataset_prompts = ["name"]
model.model_vision.dataset_names = ["lvis+stuffonly"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

train.output_dir = "output/" + __file__[:-3]
