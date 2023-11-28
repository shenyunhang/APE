from ...COCO_InstanceSegmentation.ape_deta.ape_deta_vitl_eva02_lsj1024_cp_12ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)

from ...common.data.coco_panoptic_lsj1024 import dataloader

model.model_vision.dataset_prompts = ["name"]
model.model_vision.dataset_names = ["coco_2017"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

model.model_vision.instance_on = True
model.model_vision.semantic_on = True
model.model_vision.panoptic_on = True

model.model_vision.stuff_prob_thing = -1.0

train.output_dir = "output/" + __file__[:-3]
