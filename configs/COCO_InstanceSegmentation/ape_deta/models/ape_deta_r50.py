import torch.nn as nn

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import BasicStem, ResNet
from detrex.layers import PositionEmbeddingSine
from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from ape.modeling.ape_deta import (
    DeformableCriterion,
    DeformableDETR,
    DeformableDETRSegm,
    DeformableDetrTransformer,
    DeformableDetrTransformerDecoder,
    DeformableDetrTransformerEncoder,
    SomeThing,
    Stage1Assigner,
    Stage2Assigner,
)
from ape.modeling.text import T5_warpper



model_vision = L(DeformableDETRSegm)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res2", "res3", "res4", "res5"],
        freeze_at=1,
    ),
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    neck=L(ChannelMapper)(
        input_shapes={
            "res3": ShapeSpec(channels=512),
            "res4": ShapeSpec(channels=1024),
            "res5": ShapeSpec(channels=2048),
        },
        in_features=["res3", "res4", "res5"],
        out_channels=256,
        num_outs=5,
        kernel_size=1,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
    ),
    transformer=L(DeformableDetrTransformer)(
        encoder=L(DeformableDetrTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            post_norm=False,
            num_feature_levels="${..num_feature_levels}",
        ),
        decoder=L(DeformableDetrTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels="${..num_feature_levels}",
        ),
        as_two_stage="${..as_two_stage}",
        num_feature_levels=5,
        two_stage_num_proposals="${..num_queries}",
        assign_first_stage=True,
    ),
    embed_dim=256,
    num_classes=80,
    num_queries=900,
    aux_loss=True,
    with_box_refine=True,
    as_two_stage=True,
    criterion=[
        L(DeformableCriterion)(
            num_classes="${...num_classes}",
            matcher=L(HungarianMatcher)(
                cost_class=2.0,
                cost_bbox=5.0,
                cost_giou=2.0,
                cost_class_type="focal_loss_cost",
                alpha=0.25,
                gamma=2.0,
            ),
            matcher_stage1=L(Stage1Assigner)(
                t_low=0.3,
                t_high=0.7,
                max_k=4,
            ),
            matcher_stage2=L(Stage2Assigner)(
                num_queries="${model.model_vision.num_queries}",
                num_classes="${..num_classes}",
                max_k=4,
            ),
            weight_dict={
                "loss_class": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
                "loss_mask": 5,
                "loss_dice": 5,
            },
            loss_class_type="focal_loss",
            alpha=0.25,
            gamma=2.0,
            losses=["class", "boxes", "masks"],
        ),
    ],
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    select_box_nums_for_evaluation=100,
    input_format="RGB",
    mask_encode_level=0,
    mask_in_features=["res2"],
    input_shapes={
        "res2": ShapeSpec(channels=256),
        "res3": ShapeSpec(channels=512),
        "res4": ShapeSpec(channels=1024),
        "res5": ShapeSpec(channels=2048),
    },
    output_dir=None,
    vis_period=0,
    embed_dim_language=1024,
    instance_on=True,
    semantic_on=False,
    panoptic_on=False,
)

if model_vision.aux_loss:
    for j in range(len(model_vision.criterion)):
        weight_dict = model_vision.criterion[j].weight_dict
        aux_weight_dict = {}
        for i in range(model_vision.transformer.decoder.num_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        model_vision.criterion[j].weight_dict = weight_dict

model = L(SomeThing)(
    model_vision=model_vision,
    model_language=L(T5_warpper)(
        pretrained_model_name_or_path="models/google/flan-t5-large/",
        eval_only=True,
    ),
)
