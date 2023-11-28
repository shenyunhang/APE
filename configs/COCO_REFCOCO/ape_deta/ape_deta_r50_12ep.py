from detectron2.config import LazyCall as L
from detrex.config import get_config

from ...COCO_InstanceSegmentation.ape_deta.ape_deta_r50_12ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)
from ...common.data.coco_refcoco_instance import dataloader

model.model_vision.num_classes = 80
model.model_vision.select_box_nums_for_evaluation = 300

criterion = model.model_vision.criterion[0]
model.model_vision.criterion = [criterion for _ in range(2)]
for criterion, num_classes in zip(model.model_vision.criterion, [80, 1]):
    criterion.num_classes = num_classes

model.model_vision.criterion[1].weight_dict["loss_class_enc"] = 0.0

dataloader.train.total_batch_size = 16
dataloader.train.total_batch_size_list = [16, 16]

model.model_vision.dataset_prompts = ["name", "expression"]
model.model_vision.dataset_names = ["coco_2017", "refcoco"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

train.output_dir = "output/" + __file__[:-3]
