from detectron2.config import LazyCall as L
from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation.lvis_evaluation import LVISEvaluator

from ...COCO_InstanceSegmentation.ape_deta.ape_deta_r50_12ep import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

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

model.model_vision.num_classes = 1203
model.model_vision.select_box_nums_for_evaluation = 300
model.model_vision.criterion[0].num_classes = 1203
model.model_vision.criterion[0].use_fed_loss = True
model.model_vision.criterion[0].get_fed_loss_cls_weights = lambda: get_fed_loss_cls_weights(
    dataloader.train.dataset.names, 0.5
)
model.model_vision.criterion[0].fed_loss_num_classes = 50

train.max_iter = 180000
train.eval_period = 20000

lr_multiplier.scheduler.milestones = [150000, 180000]
lr_multiplier.warmup_length = 250 / train.max_iter

model.model_vision.dataset_prompts = ["name"]
model.model_vision.dataset_names = ["lvis"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

train.output_dir = "output/" + __file__[:-3]
