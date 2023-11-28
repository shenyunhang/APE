from ...COCO_InstanceSegmentation.ape_deta.ape_deta_vitl_eva02_lsj1024_cp_12ep import (
    lr_multiplier,
    model,
    optimizer,
    train,
)
from ...common.data.pascalcontext59_semantic_lsj1024 import dataloader

model.model_vision.dataset_prompts = ["name"]
model.model_vision.name_prompt_fusion_text = [False]
model.model_vision.dataset_names = ["pascal_context_59_sem_seg"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

model.model_vision.num_classes = 59
model.model_vision.criterion[0].num_classes = 59
model.model_vision.select_box_nums_for_evaluation = 300

model.model_vision.instance_on = False
model.model_vision.semantic_on = True
model.model_vision.panoptic_on = False

model.model_vision.stuff_prob_thing = -1.0

train.output_dir = "output/" + __file__[:-3]
