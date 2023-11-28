from ...common.data.odinw35_instance import dataloader
from .ape_deta_r50_12ep import lr_multiplier, model, optimizer, train

model.model_vision.dataset_prompts = ["name" for _ in dataloader.tests]
model.model_vision.dataset_names = [
    test.dataset.names.replace("_val", "") for test in dataloader.tests
]
model.model_vision.dataset_metas = [test.dataset.names for test in dataloader.tests]

train.output_dir = "output/" + __file__[:-3]
