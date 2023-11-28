from detectron2.config import LazyCall as L
from omegaconf import OmegaConf
from ape.layers import VisionLanguageFusion
from ape.modeling.ape_deta import (
    DeformableDETRSegmVL,
    DeformableDetrTransformerDecoderVL,
    DeformableDetrTransformerEncoderVL,
    DeformableDetrTransformerVL,
)

from .ape_deta_r50_12ep import dataloader, lr_multiplier, model, optimizer, train

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
    cfg=OmegaConf.from_dotlist(
        [
            "MODEL.DYHEAD.FUSE_CONFIG.STABLE_SOFTMAX_2D=False",
            "MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW=True",
            "MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW=True",
            "MODEL.VL_FUSION_USE_CHECKPOINT=True",
        ],
    ),
)


model.model_vision.text_feature_bank = False
model.model_vision.text_feature_reduce_before_fusion = False
model.model_vision.text_feature_batch_repeat = False
model.model_vision.expression_cumulative_gt_class = False

train.output_dir = "output/" + __file__[:-3]
model.model_vision.vis_period = 5120
