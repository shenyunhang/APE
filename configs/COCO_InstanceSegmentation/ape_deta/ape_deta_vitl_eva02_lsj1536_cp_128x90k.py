from .ape_deta_vitl_eva02_lsj1536_cp_64x90k import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

dataloader.train.total_batch_size = 128

train.output_dir = "output/" + __file__[:-3]
