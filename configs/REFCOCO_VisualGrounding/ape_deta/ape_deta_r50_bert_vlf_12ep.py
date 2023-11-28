from detectron2.config import LazyCall as L
from ape.modeling.text import Bert

from ...common.data.refcoco_instance import dataloader
from .ape_deta_r50_vlf_12ep import lr_multiplier, model, optimizer, train

model.model_vision.num_classes = 1
model.model_vision.select_box_nums_for_evaluation = 1

model.model_vision.criterion[0].num_classes = 1
criterion = model.model_vision.criterion[0]
model.model_vision.criterion = [criterion for _ in range(2)]

model.model_vision.dataset_prompts = ["expression"]
model.model_vision.dataset_names = ["refcoco"]
model.model_vision.dataset_metas = dataloader.train.dataset.names


model.model_language = L(Bert)(
    pretrained_model_name_or_path="models/huggingface/bert-base-uncased/"
)
model.model_vision.embed_dim_language = 768
model.model_vision.text_feature_reduce_type = "average"

model.model_vision.text_feature_bank = False
model.model_vision.text_feature_reduce_before_fusion = False
model.model_vision.text_feature_batch_repeat = False
model.model_vision.expression_cumulative_gt_class = False


train.output_dir = "output/" + __file__[:-3]
model.model_vision.vis_period = 5120
