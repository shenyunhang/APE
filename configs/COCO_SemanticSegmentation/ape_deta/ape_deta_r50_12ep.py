from ...COCO_InstanceSegmentation.ape_deta.ape_deta_r50_12ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)
from ...common.data.coco_semantic import dataloader

model.model_vision.dataset_prompts = ["name"]
model.model_vision.dataset_names = ["stuffonly"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

num_classes = 54
model.model_vision.num_classes = num_classes
model.model_vision.criterion[0].num_classes = num_classes
model.model_vision.criterion[0].matcher_stage2.num_classes = num_classes

model.model_vision.instance_on = False
model.model_vision.semantic_on = True
model.model_vision.panoptic_on = False

model.model_vision.stuff_prob_thing = 0.9

model.model_vision.semantic_post_nms = False
model.model_vision.aux_mask = True

train.output_dir = "output/" + __file__[:-3]
