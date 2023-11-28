from ...common.data.lviscocococostuff_panoptic_lsj1024_cp import dataloader
from ...LVIS_InstanceSegmentation.ape_deta.ape_deta_vitl_eva02_lsj1024_cp_24ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)

model.model_vision.num_classes = 1256
model.model_vision.criterion[0].num_classes = 1256

model.model_vision.instance_on = True
model.model_vision.semantic_on = True
model.model_vision.panoptic_on = False

model.model_vision.dataset_prompts = ["name"]
model.model_vision.dataset_names = ["lvis+stuffonly"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

train.output_dir = "output/" + __file__[:-3]
