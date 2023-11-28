from .ape_deta_vitl_eva02_clip_lsj1536_cp_64x90k import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

train.max_iter = 45000

train.eval_period = 2500

train.checkpointer.period = 2500

lr_multiplier.scheduler.milestones = [37500, 45000]

dataloader.train.total_batch_size = 128

train.output_dir = "output/" + __file__[:-3]
