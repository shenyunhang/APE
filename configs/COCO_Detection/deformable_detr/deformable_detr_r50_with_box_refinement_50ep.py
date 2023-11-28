from .deformable_detr_r50_50ep import dataloader, lr_multiplier, model, optimizer, train

model.with_box_refine = True

train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/deformable_detr_with_box_refinement_50ep"
