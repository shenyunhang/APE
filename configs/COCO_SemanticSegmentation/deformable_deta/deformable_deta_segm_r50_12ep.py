from ...COCO_InstanceSegmentation.deformable_deta.deformable_deta_segm_r50_12ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)
from ...common.data.coco_semantic import dataloader

num_classes = 54
model.num_classes = num_classes
model.criterion.num_classes = num_classes
model.criterion.matcher_stage2.num_classes = num_classes

model.instance_on = False
model.semantic_on = True
model.panoptic_on = False

train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.init_checkpoint = "models/torchvision/R-50.pkl"
train.output_dir = "output/" + __file__[:-3]

model.dataset_metas = dataloader.train.dataset.names
