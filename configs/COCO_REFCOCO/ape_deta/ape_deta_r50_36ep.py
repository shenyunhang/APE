from detrex.config import get_config

from .ape_deta_r50_12ep import dataloader, model, optimizer, train

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_36ep

train.output_dir = "output/" + __file__[:-3]
model.model_vision.output_dir = train.output_dir

train.max_iter = 270000

train.eval_period = 15000
