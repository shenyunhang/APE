from detrex.config import get_config

from .deformable_deta_segm_r50_12ep import dataloader, model, optimizer, train

lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_36ep

train.max_iter = 270000

train.output_dir = "output/" + __file__[:-3]
