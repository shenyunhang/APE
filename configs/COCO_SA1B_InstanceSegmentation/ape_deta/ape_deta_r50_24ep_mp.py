from .ape_deta_r50_24ep import dataloader, lr_multiplier, model, optimizer, train

model.model_vision.transformer.proposal_ambiguous = 1

train.output_dir = "output/" + __file__[:-3]
model.model_vision.vis_period = 128
