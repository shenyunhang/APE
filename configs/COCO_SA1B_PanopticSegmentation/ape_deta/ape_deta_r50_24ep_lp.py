from .ape_deta_r50_24ep import dataloader, lr_multiplier, model, optimizer, train

model.model_vision.criterion[0].losses += ["iou"]
model.model_vision.criterion[0].weight_dict["loss_ious"] = 1.0

model.model_vision.last_class_embed_use_mlp = True

train.output_dir = "output/" + __file__[:-3]
model.model_vision.vis_period = 128
