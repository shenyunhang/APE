from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ...COCO_InstanceSegmentation.deformable_deta.deformable_deta_segm_r50_12ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)
from ...common.data.ade20k_semantic import dataloader

num_classes = 150
model.num_classes = num_classes
model.criterion.num_classes = num_classes
model.criterion.matcher_stage2.num_classes = num_classes
model.criterion.eos_coef = 1.0

model.instance_on = False
model.semantic_on = True
model.panoptic_on = False

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

train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.init_checkpoint = "models/torchvision/R-50.pkl"
train.output_dir = "output/" + __file__[:-3]

train.amp.enabled = True
train.ddp.fp16_compression = True
train.ddp.find_unused_parameters = True

model.dataset_metas = dataloader.train.dataset.names
