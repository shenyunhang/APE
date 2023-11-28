from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from .ape_deta_vitl_eva02_vlf_lsj1024_cp_720k import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

train.max_iter = 1080000
train.eval_period = 1080000

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[900000],
        num_updates=1080000,
    ),
    warmup_length=2000 / 1080000,
    warmup_method="linear",
    warmup_factor=0.001,
)

train.output_dir = "output/" + __file__[:-3]
