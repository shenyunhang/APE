from detectron2.config import LazyCall as L

from ape.data.detection_utils import get_fed_loss_cls_weights

from ...common.data.lviscocococostuff_o365_oid_vgr_sa1b_refcoco_group_by_image_panoptic_lsj1024_cp import (
    dataloader,
)
from ...LVISCOCOCOCOSTUFF_O365_OID_VGR_REFCOCO.ape_deta.ape_deta_vitl_eva02_vlf_lsj1024_cp_1080k import (
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
del criterion.fed_loss_num_classes
model.model_vision.criterion = [criterion for _ in range(7)]
for criterion, num_classes in zip(model.model_vision.criterion, [1256, 365, 601, 200, 1, 200, 200]):
    criterion.num_classes = num_classes

dataloader.train.mapper.max_num_phrase = 100
dataloader.train.mapper.nms_thresh_phrase = 0.6

model.model_vision.criterion[0].use_fed_loss = True
model.model_vision.criterion[0].get_fed_loss_cls_weights = lambda: get_fed_loss_cls_weights(
    dataloader.train.dataset.names[0], 0.5
)
model.model_vision.criterion[0].fed_loss_num_classes = 50
model.model_vision.criterion[0].fed_loss_pad_type = "cat"

model.model_vision.criterion[3].weight_dict["loss_class_enc"] = 0.0
for k, v in model.model_vision.criterion[3].weight_dict.items():
    if "_enc" in k:
        model.model_vision.criterion[3].weight_dict.update({k: 0.0})
    if "_bbox" in k or "_giou" in k:
        model.model_vision.criterion[3].weight_dict.update({k: 0.0})

for k, v in model.model_vision.criterion[4].weight_dict.items():
    if "_class" in k and "_enc" not in k:
        model.model_vision.criterion[4].weight_dict.update({k: 0.0})

model.model_vision.criterion[5].weight_dict["loss_class_enc"] = 0.0

model.model_vision.stuff_dataset_learn_thing = False
model.model_vision.stuff_prob_thing = 0.9

dataloader.train.total_batch_size = 16
dataloader.train.total_batch_size_list = [16, 16, 16, 16, 16, 16]

model.model_vision.dataset_prompts = [
    "name",
    "name",
    "name",
    "phrase",
    "name",
    "phrase",
    "expression",
]
model.model_vision.dataset_names = [
    "lvis+stuffonly",
    "objects365",
    "openimages",
    "vgregion",
    "sa1b",
    "refcoco-mixed_group-by-image",
    "refcoco",
]
model.model_vision.dataset_metas = dataloader.train.dataset.names + ["refcoco-mixed"]

train.output_dir = "output/" + __file__[:-3]
