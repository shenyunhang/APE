from ...COCO_InstanceSegmentation.deformable_deta.deformable_deta_segm_r50_12ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)

from ...common.data.coco_panoptic import dataloader

model.num_classes = 133
model.criterion.num_classes = 133
model.dataset_metas = dataloader.train.dataset.names

model.stuff_dataset_learn_thing = False

model.instance_on = True
model.semantic_on = True
model.panoptic_on = True

model.stuff_prob_thing = -1.0


model.semantic_post_nms = False
model.panoptic_post_nms = True
model.aux_mask = True

train.output_dir = "output/" + __file__[:-3]
