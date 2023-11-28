from ape.modeling.ape_deta import Stage1Assigner_loc, Stage2Assigner_loc

from .ape_deta_r50_50ep import dataloader, lr_multiplier, model, optimizer, train

model.model_vision.criterion[0].losses += ["pred_iou"]
model.model_vision.criterion[0].weight_dict["loss_iou"] = 1.0

model.model_vision.last_class_embed_use_mlp = True
model.model_vision.transformer.pre_nms_topk = 1000
model.model_vision.transformer.nms_thresh_enc = 0.9

model.model_vision.criterion[0].matcher_stage1.update(
    _target_=Stage1Assigner_loc,
)
model.model_vision.criterion[1].matcher_stage1.update(
    _target_=Stage1Assigner_loc,
)


train.output_dir = "output/" + __file__[:-3]
model.model_vision.vis_period = 1280
