from .ape_deta_r50_50ep import dataloader, lr_multiplier, model, optimizer, train

model.model_vision.transformer.proposal_ambiguous = 1
model.model_vision.transformer.encoder.use_act_checkpoint = True

train.output_dir = "output/" + __file__[:-3]
model.model_vision.vis_period = 12800
