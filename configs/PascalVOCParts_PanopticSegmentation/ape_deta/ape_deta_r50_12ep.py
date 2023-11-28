from ...COCO_InstanceSegmentation.ape_deta.ape_deta_r50_12ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)
from ...common.data.pascalvocpart_panoptic import dataloader

model.model_vision.num_classes = 200

model.model_vision.criterion[0].num_classes = 200

model.model_vision.dataset_prompts = ["name"]
model.model_vision.dataset_names = ["pascal_parts"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

train.output_dir = "output/" + __file__[:-3]
model.model_vision.vis_period = 1280
