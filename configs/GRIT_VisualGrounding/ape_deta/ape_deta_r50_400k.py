from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ...COCO_InstanceSegmentation.ape_deta.ape_deta_r50_12ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)
from ...common.data.grit_instance import dataloader

model.model_vision.num_classes = 200

criterion = model.model_vision.criterion[0]
model.model_vision.criterion = [criterion for _ in range(2)]
for criterion, num_classes in zip(model.model_vision.criterion, [200, 200]):
    criterion.num_classes = num_classes

dataloader.train.mapper.max_num_phrase = 200
dataloader.train.mapper.nms_thresh_phrase = 0.6

model.model_vision.dataset_prompts = ["phrase", "expression"]
model.model_vision.dataset_names = ["grit", "refcoco"]
model.model_vision.dataset_metas = dataloader.train.dataset.names + ["refcoco-mixed"]

train.max_iter = 400000
train.eval_period = 10000

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[333000],
        num_updates=400000,
    ),
    warmup_length=2000 / 400000,
    warmup_method="linear",
    warmup_factor=0.001,
)

train.output_dir = "output/" + __file__[:-3]
model.model_vision.vis_period = 1280
