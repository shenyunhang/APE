import random

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ape.data.detection_utils import get_fed_loss_cls_weights
from ape.data.samplers import MultiDatasetTrainingSampler

from ...common.data.lviscocococostuff_o365_oid_vg_panoptic_lsj1024_cp import dataloader
from ...LVIS_InstanceSegmentation.ape_deta.ape_deta_vitl_eva02_lsj1024_cp_24ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)

model.model_vision.num_classes = 1256
model.model_vision.select_box_nums_for_evaluation = 300

criterion = model.model_vision.criterion[0]
del criterion.use_fed_loss
del criterion.get_fed_loss_cls_weights
model.model_vision.criterion = [criterion for _ in range(4)]
for criterion, num_classes in zip(model.model_vision.criterion, [1256, 365, 601, 150]):
    criterion.num_classes = num_classes

dataloader.train.mapper.max_num_phrase = 100
dataloader.train.mapper.nms_thresh_phrase = 0.8

model.model_vision.criterion[0].use_fed_loss = True
model.model_vision.criterion[0].get_fed_loss_cls_weights = lambda: get_fed_loss_cls_weights(
    dataloader.train.dataset.names[0], 0.5
)
model.model_vision.criterion[0].fed_loss_num_classes = 50

model.model_vision.stuff_dataset_learn_thing = False
model.model_vision.stuff_prob_thing = 0.9

model.model_vision.instance_on = True
model.model_vision.semantic_on = True
model.model_vision.panoptic_on = False

model.model_vision.neck = None

train.max_iter = 720000
train.eval_period = 720000

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[640000],
        num_updates=720000,
    ),
    warmup_length=1000 / 720000,
    warmup_method="linear",
    warmup_factor=0.001,
)

dataloader.train.total_batch_size = 16
dataloader.train.total_batch_size_list = [16, 16, 16, 16]

model.model_vision.dataset_prompts = ["name", "name", "name", "name"]
model.model_vision.dataset_names = [
    "lvis+stuffonly",
    "objects365",
    "openimages",
    "visualgenome",
]
model.model_vision.dataset_metas = dataloader.train.dataset.names

train.output_dir = "output/" + __file__[:-3]

dataloader.train.sampler = lambda dataset_dicts: MultiDatasetTrainingSampler(
    repeat_factors=MultiDatasetTrainingSampler.get_repeat_factors(
        dataset_dicts=dataset_dicts,
        num_datasets=4,
        dataset_ratio=[1, 1, 1, 1],
        use_rfs=[True, True, True, False],
        use_cas=[False, False, False, False],
        repeat_thresh=0.001,
        cas_lambda=1.0,
    ),
    seed=random.randint(0, 2**31),
)
