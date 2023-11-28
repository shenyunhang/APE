from ....configs.common.data.lvis_lsj1024_cp import dataloader
from .deformable_deta_vitg_eva_two_stage_lsj1024_24ep import lr_multiplier, model, optimizer, train

train.amp.enabled = True
train.ddp.fp16_compression = True

model.backbone.net.use_act_checkpoint = True
model.backbone.net.frozen_stages = 25

train.output_dir = "output/" + __file__[:-3]
dataloader.evaluator.output_dir = train.output_dir
dataloader.train.mapper.output_dir = train.output_dir

optimizer.lr = 1e-4
