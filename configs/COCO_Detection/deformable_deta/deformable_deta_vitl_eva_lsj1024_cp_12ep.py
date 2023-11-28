from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from ...common.data.coco_lsj1024_cp import dataloader
from .deformable_deta_vitl_lsj1024_12ep import lr_multiplier, model, optimizer, train

train.init_checkpoint = "models/BAAI/EVA/eva_l_psz14to16.pt?matching_heuristics=True"

optimizer.params.lr_factor_func = (
    lambda module_name: 0.1
    if "reference_points" in module_name or "sampling_offsets" in module_name
    else get_vit_lr_decay_rate(module_name, lr_decay_rate=0.8, num_layers=24)
    if "backbone" in module_name
    else 1
)

optimizer.lr = 2e-4
optimizer.weight_decay = 1e-4

train.amp.enabled = True
train.ddp.fp16_compression = True
model.backbone.net.use_act_checkpoint = False

train.output_dir = "output/" + __file__[:-3]
dataloader.evaluator.output_dir = train.output_dir
dataloader.train.mapper.output_dir = train.output_dir
