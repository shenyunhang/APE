from detectron2.config import LazyCall as L
from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation.lvis_evaluation import LVISEvaluator
from detrex.config import get_config

from .....detectron2.projects.ViTDet.configs.common.coco_loader_lsj import dataloader
from ...COCO_Detection.deformable_deta.deformable_deta_r50_two_stage_12ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)

dataloader.train.mapper.image_format = "BGR"

dataloader.train.total_batch_size = 16

dataloader.train.dataset.names = "lvis_v1_train"
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001
    )
)
dataloader.test.dataset.names = "lvis_v1_val"
dataloader.evaluator = L(LVISEvaluator)(
    dataset_name="${..test.dataset.names}",
    max_dets_per_image=300,
)

model.num_classes = 1203
model.criterion.num_classes = 1203
model.select_box_nums_for_evaluation = 300
model.criterion.use_fed_loss = True
model.criterion.get_fed_loss_cls_weights = lambda: get_fed_loss_cls_weights(
    dataloader.train.dataset.names, 0.5
)
model.criterion.fed_loss_num_classes = 50

train.max_iter = 180000
train.eval_period = 20000

lr_multiplier.scheduler.milestones = [150000, 180000]
lr_multiplier.warmup_length = 250 / train.max_iter

train.output_dir = "output/" + __file__[:-3]
dataloader.evaluator.output_dir = train.output_dir
