from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detrex.config import get_config

from ...COCO_SA1B_InstanceSegmentation.ape_deta.ape_deta_r50_24ep import model, optimizer, train

from ...common.data.lvis_sa1b_instance import dataloader

model.model_vision.num_classes = 1203
model.model_vision.criterion[0].num_classes = 1203
model.model_vision.select_box_nums_for_evaluation = 300
model.model_vision.criterion[0].use_fed_loss = True
model.model_vision.criterion[0].get_fed_loss_cls_weights = lambda: get_fed_loss_cls_weights(
    dataloader.train.dataset.names[0], 0.5
)
model.model_vision.criterion[0].fed_loss_num_classes = 50

model.model_vision.semantic_on = False
model.model_vision.panoptic_on = False

train.max_iter = 375000
train.eval_period = 20000

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep

model.model_vision.dataset_prompts = ["name", "name"]
model.model_vision.dataset_names = ["lvis", "sa1b"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

train.output_dir = "output/" + __file__[:-3]
