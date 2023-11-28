from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from .deformable_deta_vitb_lsj1024_12ep import dataloader, lr_multiplier, model, optimizer, train

model.backbone.net.embed_dim = 1024
model.backbone.net.depth = 24
model.backbone.net.num_heads = 16
model.backbone.net.drop_path_rate = 0.4
model.backbone.net.window_block_indexes = (
    list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
)

optimizer.params.lr_factor_func = (
    lambda module_name: 0.1
    if "reference_points" in module_name or "sampling_offsets" in module_name
    else get_vit_lr_decay_rate(module_name, lr_decay_rate=0.8, num_layers=24)
    if "backbone" in module_name
    else 1
)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}

optimizer.lr = 2e-4
optimizer.weight_decay = 0.05

train.init_checkpoint = (
    "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_large.pth?matching_heuristics=True"
)
train.init_checkpoint = "models/MAE/mae_pretrain_vit_large.pth?matching_heuristics=True"

train.amp.enabled = True
train.ddp.fp16_compression = True
model.backbone.net.use_act_checkpoint = False

train.output_dir = "output/" + __file__[:-3]
dataloader.evaluator.output_dir = train.output_dir
