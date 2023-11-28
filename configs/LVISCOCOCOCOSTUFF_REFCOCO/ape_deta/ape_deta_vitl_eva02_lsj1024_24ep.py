from detectron2.data.detection_utils import get_fed_loss_cls_weights

from ...common.data.lviscocococostuff_refcoco_panoptic_lsj1024 import dataloader
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
model.model_vision.criterion = [criterion for _ in range(2)]
for criterion, num_classes in zip(model.model_vision.criterion, [1256, 200]):
    criterion.num_classes = num_classes

model.model_vision.criterion[1].weight_dict["loss_class_enc"] = 0.0

model.model_vision.instance_on = True
model.model_vision.semantic_on = True
model.model_vision.panoptic_on = False

model.model_vision.criterion[0].use_fed_loss = True
model.model_vision.criterion[0].get_fed_loss_cls_weights = lambda: get_fed_loss_cls_weights(
    dataloader.train.dataset.names[0], 0.5
)
model.model_vision.criterion[0].fed_loss_num_classes = 50
model.model_vision.criterion[0].fed_loss_pad_type = "cat"

model.model_vision.neck = None

dataloader.train.total_batch_size = 16
dataloader.train.total_batch_size_list = [16, 16]

model.model_vision.dataset_prompts = ["name", "expression"]
model.model_vision.dataset_names = ["lvis+stuff", "refcoco"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

train.output_dir = "output/" + __file__[:-3]
