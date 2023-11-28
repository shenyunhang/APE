from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ...COCO_InstanceSegmentation.ape_deta.ape_deta_r50_12ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)
from ...common.data.ade20k_panoptic import dataloader

num_classes = 150
model.num_classes = num_classes
model.criterion.num_classes = num_classes
model.criterion.matcher_stage2.num_classes = num_classes

model.model_vision.dataset_prompts = ["name"]
model.model_vision.dataset_names = ["ade20k_panoptic"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

model.model_vision.instance_on = True
model.model_vision.semantic_on = True
model.model_vision.panoptic_on = True

model.model_vision.stuff_prob_thing = -1.0

train.max_iter = 160000
train.eval_period = 5000

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[135000, 150000],
        num_updates=160000,
    ),
    warmup_length=1000 / 160000,
    warmup_method="linear",
    warmup_factor=0.001,
)

model.model_vision.semantic_post_nms = False
model.model_vision.panoptic_post_nms = True
model.model_vision.aux_mask = True

train.output_dir = "output/" + __file__[:-3]
