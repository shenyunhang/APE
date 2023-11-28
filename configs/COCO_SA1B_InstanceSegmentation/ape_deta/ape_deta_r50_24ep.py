from detrex.config import get_config

from ...COCO_InstanceSegmentation.ape_deta.ape_deta_r50_12ep import model, optimizer, train
from ...common.data.coco_sa1b_instance import dataloader

model.model_vision.num_classes = 80
criterion = model.model_vision.criterion[0]
model.model_vision.criterion = [criterion for _ in range(2)]
for criterion, num_classes in zip(model.model_vision.criterion, [80, 1]):
    criterion.num_classes = num_classes

for k, v in model.model_vision.criterion[1].weight_dict.items():
    if "_class" in k and "_enc" not in k:
        model.model_vision.criterion[1].weight_dict.update({k: 0.0})

model.model_vision.stuff_dataset_learn_thing = False
model.model_vision.stuff_prob_thing = 0.9

model.model_vision.instance_on = True
model.model_vision.semantic_on = False
model.model_vision.panoptic_on = False

train.max_iter = 180000
train.eval_period = 5000

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_24ep

dataloader.train.total_batch_size = 16
dataloader.train.total_batch_size_list = [16, 16]

model.model_vision.dataset_prompts = ["name", "name"]
model.model_vision.dataset_names = ["coco", "sa1b"]
model.model_vision.dataset_metas = dataloader.train.dataset.names

train.output_dir = "output/" + __file__[:-3]
model.model_vision.vis_period = 1280
