from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ape.data.detection_utils import get_fed_loss_cls_weights
from ape.layers import VisionLanguageFusion
from ape.modeling.ape_deta import (
    DeformableDETRSegmVL,
    DeformableDetrTransformerDecoderVL,
    DeformableDetrTransformerEncoderVL,
    DeformableDetrTransformerVL,
)

from ...common.data.roboflow100_instance_lsj1024 import dataloader
from ...LVIS_InstanceSegmentation.ape_deta.ape_deta_vitl_eva02_vlf_lsj1024_cp_24ep import (
    model,
    optimizer,
    train,
)

model.model_vision.num_classes = 1256
model.model_vision.select_box_nums_for_evaluation = 300

criterion = model.model_vision.criterion[0]
del criterion.use_fed_loss
del criterion.get_fed_loss_cls_weights
model.model_vision.criterion = [criterion for _ in range(100)]
for criterion, num_classes in zip(
    model.model_vision.criterion,
    [
        1000,
    ]
    * 100,
):
    criterion.num_classes = num_classes

model.model_vision.instance_on = True
model.model_vision.semantic_on = False
model.model_vision.panoptic_on = False

model.model_vision.neck = None

train.max_iter = 720000
train.eval_period = 720000

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[640000],
        num_updates=720000,
    ),
    warmup_length=1000 / 720000,
    warmup_method="linear",
    warmup_factor=0.001,
)


model.model_vision.dataset_prompts = ["name" for _ in dataloader.tests]
model.model_vision.dataset_names = [x.dataset.names for x in dataloader.tests]
model.model_vision.dataset_metas = [x.dataset.names for x in dataloader.tests]
model.model_vision.name_prompt_fusion_text = dataloader.name_prompt_fusion_text
model.model_vision.select_box_nums_for_evaluation_list = (
    dataloader.select_box_nums_for_evaluation_list
)


model.model_vision.update(
    _target_=DeformableDETRSegmVL,
)
model.model_vision.transformer.update(
    _target_=DeformableDetrTransformerVL,
)
model.model_vision.transformer.encoder.update(
    _target_=DeformableDetrTransformerEncoderVL,
)
model.model_vision.transformer.decoder.update(
    _target_=DeformableDetrTransformerDecoderVL,
)


model.model_vision.transformer.encoder.vl_layer = L(VisionLanguageFusion)(
    v_dim="${....embed_dim}",
    l_dim="${....embed_dim_language}",
    embed_dim=2048,
    num_heads=8,
    dropout=0.1,
    drop_path=0.0,
    init_values=1.0 / 6,
    stable_softmax_2d=True,
    clamp_min_for_underflow=True,
    clamp_max_for_overflow=True,
    use_checkpoint=True,
)

model.model_vision.text_feature_bank = True
model.model_vision.text_feature_reduce_before_fusion = True
model.model_vision.text_feature_batch_repeat = True
model.model_vision.expression_cumulative_gt_class = True
model.model_vision.name_prompt_fusion_type = "zero"

train.output_dir = "output/" + __file__[:-3]
