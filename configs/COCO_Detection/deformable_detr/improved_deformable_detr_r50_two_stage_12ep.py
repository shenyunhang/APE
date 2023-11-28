from .improved_deformable_detr_r50_12ep import dataloader, lr_multiplier, model, optimizer, train

model.with_box_refine = True
model.as_two_stage = True

train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/improved_deformable_detr_r50_two_stage_12ep"
