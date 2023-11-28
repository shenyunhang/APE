from ...COCO_InstanceSegmentation.ape_deta.ape_deta_r50_12ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)

from ...common.data.coco_panoptic_separated import dataloader

model.model_vision.dataset_prompts = ["name"]
model.model_vision.dataset_names = ["coco_2017"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

model.model_vision.instance_on = True
model.model_vision.semantic_on = True
model.model_vision.panoptic_on = True

model.model_vision.stuff_prob_thing = -1.0

model.model_vision.semantic_post_nms = False
model.model_vision.panoptic_post_nms = True
model.model_vision.aux_mask = True

train.output_dir = "output/" + __file__[:-3]
