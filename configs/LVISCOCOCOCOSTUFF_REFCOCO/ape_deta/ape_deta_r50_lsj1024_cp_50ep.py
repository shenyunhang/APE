from ...common.data.lviscocococostuff_refcoco_panoptic_lsj1024_cp import dataloader
from .ape_deta_r50_lsj1024_50ep import dataloader, lr_multiplier, model, optimizer, train

train.output_dir = "output/" + __file__[:-3]
