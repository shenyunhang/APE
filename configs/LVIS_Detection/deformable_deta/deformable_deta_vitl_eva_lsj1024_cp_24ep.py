from ...common.data.lvis_lsj1024_cp import dataloader
from .deformable_deta_vitl_two_stage_lsj1024_24ep import lr_multiplier, model, optimizer, train

train.init_checkpoint = "models/BAAI/EVA/eva_l_psz14to16.pt?matching_heuristics=True"

train.amp.enabled = True
train.ddp.fp16_compression = True

model.backbone.net.use_act_checkpoint = False

train.output_dir = "output/" + __file__[:-3]
dataloader.evaluator.output_dir = train.output_dir
dataloader.train.mapper.output_dir = train.output_dir
