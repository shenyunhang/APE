from detectron2.config import LazyCall as L
from ape.modeling.text import Bert

from .ape_deta_r50_vlf_lsj1024_cp_50ep import dataloader, lr_multiplier, model, optimizer, train

model.model_vision.criterion[1].num_classes = 1

model.model_language = L(Bert)(
    pretrained_model_name_or_path="models/huggingface/bert-base-uncased/"
)
model.model_vision.embed_dim_language = 768
model.model_vision.text_feature_reduce_type = "average"

model.model_vision.text_feature_bank = False
model.model_vision.text_feature_reduce_before_fusion = False
model.model_vision.text_feature_batch_repeat = False
model.model_vision.expression_cumulative_gt_class = False
model.model_vision.name_prompt_fusion_type = "none"

train.output_dir = "output/" + __file__[:-3]
model.model_vision.vis_period = 5120
