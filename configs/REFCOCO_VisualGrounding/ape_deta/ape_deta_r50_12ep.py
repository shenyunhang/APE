from ...COCO_InstanceSegmentation.ape_deta.ape_deta_r50_12ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)

from ...common.data.refcoco_group_by_image_instance import dataloader

model.model_vision.num_classes = 256

model.model_vision.select_box_nums_for_evaluation = 1

criterion = model.model_vision.criterion[0]
model.model_vision.criterion = [criterion for _ in range(2)]

model.model_vision.dataset_prompts = ["phrase", "expression"]
model.model_vision.dataset_names = ["refcoco-mixed_group-by-image", "refcoco"]
model.model_vision.dataset_metas = dataloader.train.dataset.names + ["refcoco-mixed"]

model.model_vision.text_feature_bank = True
model.model_vision.text_feature_reduce_before_fusion = True
model.model_vision.text_feature_batch_repeat = True
model.model_vision.expression_cumulative_gt_class = True

train.output_dir = "output/" + __file__[:-3]
model.model_vision.vis_period = 5120
