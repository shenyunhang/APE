from ...COCO_InstanceSegmentation.ape_deta.ape_deta_r50_12ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)
from ...common.data.flickr30k_instance import dataloader

model.model_vision.num_classes = 200

model.model_vision.criterion[0].num_classes = 200

dataloader.train.mapper.max_num_phrase = 100

model.model_vision.dataset_prompts = ["phrase", "expression"]
model.model_vision.dataset_names = ["flickr30k", "refcoco"]
model.model_vision.dataset_metas = dataloader.train.dataset.names + ["refcoco-mixed"]

train.output_dir = "output/" + __file__[:-3]
model.model_vision.vis_period = 1280
