

from ...common.data.coco_lsj1024_cp import dataloader
from ...common.data.constants import constants
from .deformable_deta_vitb_lsj1024_12ep import lr_multiplier, model, optimizer, train

model.pixel_mean = constants.openai_imagenet_rgb256_mean
model.pixel_std = constants.openai_imagenet_rgb256_std
model.input_format = "RGB"
dataloader.train.mapper.image_format = "RGB"




train.init_checkpoint = "models/CLIP/ViT-B-16.pt"

train.output_dir = "output/" + __file__[:-3]
dataloader.evaluator.output_dir = train.output_dir
dataloader.train.mapper.output_dir = train.output_dir
dataloader.train.mapper.vis_period = 1
