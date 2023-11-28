import copy
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import Conv2d, ShapeSpec, get_norm, move_device_like
from detectron2.modeling import GeneralizedRCNN
from detectron2.modeling.meta_arch.panoptic_fpn import combine_semantic_and_instance_outputs
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.structures import BitMasks, Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid
from torchvision.ops.boxes import batched_nms

from .deformable_detr import DeformableDETR
from .segmentation import MaskHeadSmallConv, MHAttentionMap


class DeformableDETRSegm(DeformableDETR):
    """Implements the Deformable DETR model.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        backbone (nn.Module): the backbone module.
        position_embedding (nn.Module): the position embedding module.
        neck (nn.Module): the neck module.
        transformer (nn.Module): the transformer module.
        embed_dim (int): the dimension of the embedding.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): whether to use auxiliary loss. Default: True.
        with_box_refine (bool): whether to use box refinement. Default: False.
        as_two_stage (bool): whether to use two-stage. Default: False.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 100.

    """

    def __init__(
        self,
        instance_on: bool = True,
        semantic_on: bool = False,
        panoptic_on: bool = False,
        freeze_detr=False,
        input_shapes=[],
        mask_in_features=[],
        mask_encode_level=0,
        stuff_dataset_learn_thing: bool = True,
        stuff_prob_thing: float = -1.0,
        test_mask_on: bool = True,
        semantic_post_nms: bool = True,
        panoptic_post_nms: bool = True,
        aux_mask: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.instance_on = instance_on
        self.semantic_on = semantic_on
        self.panoptic_on = panoptic_on

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        self.input_shapes = input_shapes
        self.mask_in_features = mask_in_features
        self.mask_encode_level = mask_encode_level

        hidden_dim = self.transformer.embed_dim
        norm = "GN"
        use_bias = False

        assert len(self.mask_in_features) == 1
        in_channels = [self.input_shapes[feat_name].channels for feat_name in self.mask_in_features]
        in_channel = in_channels[0]

        self.lateral_conv = Conv2d(
            in_channel,
            hidden_dim,
            kernel_size=1,
            stride=1,
            bias=use_bias,
            padding=0,
            norm=get_norm(norm, hidden_dim),
        )
        self.output_conv = Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            bias=use_bias,
            padding=1,
            norm=get_norm(norm, hidden_dim),
            activation=F.relu,
        )
        self.mask_conv = Conv2d(
            hidden_dim, hidden_dim, kernel_size=1, stride=1, bias=use_bias, padding=0
        )

        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.aux_mask = aux_mask
        if self.aux_mask:
            self.mask_embed = nn.ModuleList(
                [copy.deepcopy(self.mask_embed) for i in range(len(self.class_embed) - 1)]
            )

        weight_init.c2_xavier_fill(self.lateral_conv)
        weight_init.c2_xavier_fill(self.output_conv)
        weight_init.c2_xavier_fill(self.mask_conv)

        self.stuff_dataset_learn_thing = stuff_dataset_learn_thing
        self.stuff_prob_thing = stuff_prob_thing
        self.test_mask_on = test_mask_on
        self.semantic_post_nms = semantic_post_nms
        self.panoptic_post_nms = panoptic_post_nms

    def forward(self, batched_inputs, do_postprocess=True):
        images = self.preprocess_image(batched_inputs)

        batch_size, _, H, W = images.tensor.shape
        img_masks = images.tensor.new_ones(batch_size, H, W)
        for image_id, image_size in enumerate(images.image_sizes):
            img_masks[image_id, : image_size[0], : image_size[1]] = 0

        features = self.backbone(images.tensor)  # output feature dict

        if self.neck is not None:
            multi_level_feats = self.neck({f: features[f] for f in self.neck.in_features})
        else:
            multi_level_feats = [feat for feat_name, feat in features.items()]
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(
                self.position_embedding(multi_level_masks[-1]).to(images.tensor.dtype)
            )

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight

        (
            inter_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            anchors,
            memory,
        ) = self.transformer(
            multi_level_feats, multi_level_masks, multi_level_position_embeddings, query_embeds
        )

        mask_features = self.maskdino_mask_features(memory, features, multi_level_masks)

        outputs_classes = []
        outputs_coords = []
        outputs_masks = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

            if self.aux_mask:
                mask_embeds = self.mask_embed[lvl](inter_states[lvl])
            else:
                mask_embeds = self.mask_embed(inter_states[lvl])
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embeds, mask_features)
            outputs_masks.append(outputs_mask)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_mask = outputs_masks
        if self.aux_mask:
            outputs_mask[-1] += 0.0 * sum(outputs_mask)

        output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_masks": outputs_mask[-1],
            "init_reference": init_reference,
        }
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(
                outputs_class,
                outputs_coord,
                outputs_mask,
            )

        if self.as_two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            output["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
                "anchors": anchors,
            }

        if (
            self.vis_period > 0
            and self.training
            and get_event_storage().iter % self.vis_period == self.vis_period - 1
        ):
            self.visualize_training(batched_inputs, output, images)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:

            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"]

            iter_func = retry_if_cuda_oom(F.interpolate)
            mask_pred = iter_func(
                mask_pred, size=images.tensor.size()[2:], mode="bilinear", align_corners=False
            )

            merged_results = [{} for _ in range(box_cls.size(0))]
            if self.instance_on:
                if self.metadata is not None:
                    if is_thing_stuff_overlap(self.metadata):
                        thing_id = self.metadata.thing_dataset_id_to_contiguous_id.values()
                        thing_id = torch.Tensor(list(thing_id)).to(torch.long).to(self.device)

                        detector_box_cls = torch.zeros_like(box_cls)
                        detector_box_cls += float("-inf")
                        detector_box_cls[..., thing_id] = box_cls[..., thing_id]
                    else:
                        num_thing_classes = len(self.metadata.thing_classes)
                        detector_box_cls = box_cls[..., :num_thing_classes]
                else:
                    detector_box_cls = box_cls

                detector_results, filter_inds = self.inference(
                    detector_box_cls, box_pred, images.image_sizes
                )

                if self.test_mask_on:
                    detector_mask_preds = [
                        x[filter_ind] for x, filter_ind in zip(mask_pred, filter_inds)
                    ]

                    for result, box_mask in zip(detector_results, detector_mask_preds):
                        box_mask = box_mask.sigmoid() > 0.5
                        box_mask = BitMasks(box_mask).crop_and_resize(
                            result.pred_boxes.tensor.to(box_mask.device), 128
                        )
                        result.pred_masks = (
                            box_mask.to(result.pred_boxes.tensor.device)
                            .unsqueeze(1)
                            .to(dtype=torch.float32)
                        )

                if do_postprocess:
                    assert (
                        not torch.jit.is_scripting()
                    ), "Scripting is not supported for postprocess."
                    detector_results = DeformableDETRSegm._postprocess_instance(
                        detector_results, batched_inputs, images.image_sizes
                    )
                    for merged_result, detector_result in zip(merged_results, detector_results):
                        merged_result.update(detector_result)

            else:
                detector_results = None

            if self.semantic_on:

                semantic_mask_pred = mask_pred.clone()

                if self.metadata is not None:
                    if is_thing_stuff_overlap(self.metadata):
                        semantic_box_cls = box_cls.clone()

                    else:
                        num_thing_classes = len(self.metadata.get("thing_classes", ["things"]))

                        semantic_box_cls_0 = box_cls[..., :num_thing_classes]
                        semantic_box_cls_1 = box_cls[..., num_thing_classes:]
                        semantic_box_cls_0, _ = semantic_box_cls_0.min(dim=2, keepdim=True)
                        semantic_box_cls = torch.cat(
                            [semantic_box_cls_0, semantic_box_cls_1], dim=2
                        )
                else:
                    semantic_box_cls = box_cls.clone()

                if self.semantic_post_nms:
                    _, filter_inds = self.inference(semantic_box_cls, box_pred, images.image_sizes)
                    semantic_box_cls = torch.stack(
                        [x[filter_ind] for x, filter_ind in zip(semantic_box_cls, filter_inds)],
                        dim=0,
                    )
                    semantic_mask_pred = torch.stack(
                        [x[filter_ind] for x, filter_ind in zip(semantic_mask_pred, filter_inds)],
                        dim=0,
                    )

                if do_postprocess:
                    assert (
                        not torch.jit.is_scripting()
                    ), "Scripting is not supported for postprocess."
                    semantic_results = DeformableDETRSegm._postprocess_semantic(
                        semantic_box_cls, semantic_mask_pred, batched_inputs, images
                    )
                    for merged_result, semantic_result in zip(merged_results, semantic_results):
                        if self.stuff_prob_thing > 0:
                            semantic_result["sem_seg"][0, ...] = math.log(
                                self.stuff_prob_thing / (1 - self.stuff_prob_thing)
                            )
                        merged_result.update(semantic_result)

            else:
                semantic_results = None

            if self.panoptic_on:
                assert self.metadata is not None
                if do_postprocess:
                    assert (
                        not torch.jit.is_scripting()
                    ), "Scripting is not supported for postprocess."
                    if True:
                        if self.panoptic_post_nms:
                            _, filter_inds = self.inference(box_cls, box_pred, images.image_sizes)
                            panoptic_mask_pred = [
                                x[filter_ind] for x, filter_ind in zip(mask_pred, filter_inds)
                            ]
                            panoptic_box_cls = [
                                x[filter_ind] for x, filter_ind in zip(box_cls, filter_inds)
                            ]

                        panoptic_results = DeformableDETRSegm._postprocess_panoptic(
                            panoptic_box_cls,
                            panoptic_mask_pred,
                            batched_inputs,
                            images,
                            self.metadata,
                        )
                    else:
                        panoptic_results = []
                        self.combine_overlap_thresh = 0.5
                        self.combine_stuff_area_thresh = 4096
                        self.combine_instances_score_thresh = 0.5
                        for detector_result, semantic_result in zip(
                            detector_results, semantic_results
                        ):
                            detector_r = detector_result["instances"]
                            sem_seg_r = semantic_result["sem_seg"]
                            panoptic_r = combine_semantic_and_instance_outputs(
                                detector_r,
                                sem_seg_r.argmax(dim=0),
                                self.combine_overlap_thresh,
                                self.combine_stuff_area_thresh,
                                self.combine_instances_score_thresh,
                            )
                            panoptic_results.append({"panoptic_seg": panoptic_r})
                    for merged_result, panoptic_result in zip(merged_results, panoptic_results):
                        merged_result.update(panoptic_result)

            else:
                panoptic_results = None

            if do_postprocess:
                return merged_results

            return detector_results, semantic_results, panoptic_results

    def maskdino_mask_features(self, encode_feats, multi_level_feats, multi_level_masks):
        start_idx = sum(
            [mask.shape[1] * mask.shape[2] for mask in multi_level_masks[: self.mask_encode_level]]
        )
        end_idx = sum(
            [
                mask.shape[1] * mask.shape[2]
                for mask in multi_level_masks[: self.mask_encode_level + 1]
            ]
        )
        b, h, w = multi_level_masks[self.mask_encode_level].size()

        encode_feats = encode_feats[:, start_idx:end_idx, :]
        encode_feats = encode_feats.permute(0, 2, 1).reshape(b, -1, h, w)

        x = [multi_level_feats[f] for f in self.mask_in_features]
        x = x[0]
        x = self.lateral_conv(x)
        x = x + F.interpolate(encode_feats, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.output_conv(x)
        mask_features = self.mask_conv(x)

        return mask_features

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask):
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_masks": c}
            for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1])
        ]

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """

        if True:

            scores = torch.cat(
                (
                    box_cls.sigmoid(),
                    torch.zeros((box_cls.size(0), box_cls.size(1), 1), device=self.device),
                ),
                dim=2,
            )

            boxes = box_cxcywh_to_xyxy(box_pred)

            img_h, img_w = torch.tensor(image_sizes, device=self.device).unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct[:, None, :]

            boxes = boxes.unbind(0)
            scores = scores.unbind(0)
            image_shapes = image_sizes

            results, filter_inds = fast_rcnn_inference(
                boxes,
                scores,
                image_shapes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_topk_per_image,
            )

            return results, filter_inds

        assert len(box_cls) == len(image_sizes)
        results = []

        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results, topk_indexes

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)

            if not targets_per_image.has("gt_masks"):
                gt_masks = torch.zeros((0, h, w), dtype=torch.bool)
            else:
                gt_masks = targets_per_image.gt_masks

            if not isinstance(gt_masks, torch.Tensor):
                if isinstance(gt_masks, BitMasks):
                    gt_masks = gt_masks.tensor
                else:
                    gt_masks = BitMasks.from_polygon_masks(gt_masks, h, w).tensor

            gt_masks = self._move_to_current_device(gt_masks)
            gt_masks = ImageList.from_tensors(
                [gt_masks],
                self.backbone.size_divisibility,
                padding_constraints=self.backbone.padding_constraints,
            ).tensor.squeeze(0)

            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "masks": gt_masks})

            if targets_per_image.has("is_thing"):
                new_targets[-1]["is_thing"] = targets_per_image.is_thing

        return new_targets

    def preprocess_image(self, batched_inputs):
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [x.to(self.pixel_mean.dtype) for x in images]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess_instance(
        instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes
    ):
        """
        Rescale the output instances to the target size.
        """
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r.to("cpu")})
        return processed_results

    @staticmethod
    def _postprocess_semantic(
        mask_clses,
        mask_preds,
        batched_inputs: List[Dict[str, torch.Tensor]],
        images,
        pano_temp=0.06,
        transform_eval=True,
    ):
        processed_results = []
        for mask_cls, mask_pred, input_per_image, image_size in zip(
            mask_clses, mask_preds, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            T = pano_temp
            mask_cls = mask_cls.sigmoid()

            if transform_eval:
                mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
            mask_pred = mask_pred.sigmoid()
            result = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results

    @staticmethod
    def _postprocess_panoptic(
        mask_clses,
        mask_preds,
        batched_inputs: List[Dict[str, torch.Tensor]],
        images,
        metadata,
        prob=0.5,
        pano_temp=0.06,
        transform_eval=True,
        object_mask_threshold=0.25,
        overlap_threshold=0.8,
    ):
        num_classes = len(metadata.thing_classes) + len(metadata.stuff_classes) - 1

        object_mask_threshold = 0.01
        overlap_threshold = 0.4
        prob = 0.1

        processed_results = []
        for mask_cls, mask_pred, input_per_image, image_size in zip(
            mask_clses, mask_preds, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            mask_pred = sem_seg_postprocess(mask_pred, image_size, height, width)

            T = pano_temp
            scores, labels = mask_cls.sigmoid().max(-1)
            mask_pred = mask_pred.sigmoid()
            keep = labels.ne(num_classes) & (scores > object_mask_threshold)
            if transform_eval:
                scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            panoptic_seg = torch.zeros((height, width), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0

            if cur_masks.size(0) > 0:

                cur_mask_ids = cur_prob_masks.argmax(0)

            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= prob).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < overlap_threshold:
                        continue

                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    if not isthing and not is_thing_stuff_overlap(metadata):
                        pred_class = int(pred_class) - len(metadata.thing_classes) + 1

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            processed_results.append({"panoptic_seg": (panoptic_seg, segments_info)})
        return processed_results

    @torch.no_grad()
    def visualize_training(self, batched_inputs, output, images):
        if self.output_dir is None:
            return
        if self.training:
            storage = get_event_storage()
            os.makedirs(self.output_dir + "/training", exist_ok=True)
        else:
            os.makedirs(self.output_dir + "/inference", exist_ok=True)

        pred_logits = output["pred_logits"]
        pred_boxes = output["pred_boxes"]
        pred_masks = output["pred_masks"]

        thing_classes = self.metadata.get("thing_classes", [])
        stuff_classes = self.metadata.get("stuff_classes", [])
        if len(thing_classes) > 0 and len(stuff_classes) > 0 and stuff_classes[0] == "things":
            stuff_classes = stuff_classes[1:]
        if is_thing_stuff_overlap(self.metadata):
            class_names = (
                thing_classes if len(thing_classes) > len(stuff_classes) else stuff_classes
            )
        else:
            class_names = thing_classes + stuff_classes

        num_thing_classes = len(class_names)
        pred_logits = pred_logits[..., :num_thing_classes]

        if pred_masks is not None:
            pred_masks = [
                F.interpolate(
                    pred_mask.float().cpu().unsqueeze(0),
                    size=images.tensor.size()[2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                if pred_mask.size(0) > 0
                else pred_mask
                for pred_mask in pred_masks
            ]
        else:
            pred_masks = [
                torch.zeros(pred_box.size(0), image_size[0], image_size[1])
                for pred_box, image_size in zip(pred_boxes, images.image_sizes)
            ]

        if True:
            results, filter_inds = self.inference(pred_logits, pred_boxes, images.image_sizes)
            pred_masks = [
                pred_mask[filter_ind.cpu()]
                for pred_mask, filter_ind in zip(pred_masks, filter_inds)
            ]
            for result, pred_mask in zip(results, pred_masks):
                result.pred_masks = pred_mask.sigmoid() > 0.5
        else:
            results = []
            for pred_logit, pred_box, pred_mask, image_size in zip(
                pred_logits, pred_boxes, pred_masks, images.image_sizes
            ):
                result = Instances(image_size)
                result.pred_boxes = Boxes(pred_box)
                result.scores = pred_logit[:, 0]
                result.pred_classes = torch.zeros(
                    len(pred_box), dtype=torch.int64, device=pred_logit.device
                )
                result.pred_masks = pred_mask.sigmoid() > 0.5

                results.append(result)

        from detectron2.utils.visualizer import Visualizer

        for input, result in zip(batched_inputs, results):

            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)

            if "instances" in input:
                labels = [
                    "{}".format(class_names[gt_class]) for gt_class in input["instances"].gt_classes
                ]
                v_gt = v_gt.overlay_instances(
                    boxes=input["instances"].gt_boxes,
                    masks=input["instances"].gt_masks
                    if input["instances"].has("gt_masks")
                    else None,
                    labels=labels,
                )
            else:
                v_gt = v_gt.output
            anno_img = v_gt.get_image()

            labels = [
                "{}_{:.0f}%".format(class_names[pred_class], score * 100)
                for pred_class, score in zip(result.pred_classes.cpu(), result.scores.cpu())
            ]
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=result.pred_boxes.tensor.clone().detach().cpu().numpy(),
                labels=labels,
                masks=result.pred_masks[:, : img.shape[0], : img.shape[1]]
                .clone()
                .detach()
                .cpu()
                .numpy()
                if result.has("pred_masks")
                else None,
            )
            pred_img = v_pred.get_image()

            vis_img = np.concatenate((anno_img, pred_img), axis=1)

            basename = os.path.basename(input["file_name"])
            if self.training:
                cv2.imwrite(
                    os.path.join(self.output_dir, "training", str(storage.iter) + "_" + basename),
                    vis_img[:, :, ::-1],
                )
            else:
                cv2.imwrite(
                    os.path.join(self.output_dir, "inference", basename), vis_img[:, :, ::-1]
                )

    @torch.no_grad()
    def visualize_inference_panoptic(self, batched_inputs, results):
        if self.output_dir is None:
            return
        if self.training:
            storage = get_event_storage()
            os.makedirs(self.output_dir + "/training", exist_ok=True)
        else:
            os.makedirs(self.output_dir + "/inference", exist_ok=True)

        from detectron2.utils.visualizer import Visualizer

        for input, result in zip(batched_inputs, results):

            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)

            height = input["height"]
            width = input["width"]
            img = cv2.resize(img, (width, height))

            v_gt = Visualizer(img, self.metadata)

            if "instances" in input:
                labels = [
                    "{}".format(class_names[gt_class]) for gt_class in input["instances"].gt_classes
                ]
                v_gt = v_gt.overlay_instances(
                    boxes=input["instances"].gt_boxes,
                    masks=input["instances"].gt_masks
                    if input["instances"].has("gt_masks")
                    else None,
                    labels=labels,
                )
            else:
                v_gt = v_gt.output
            anno_img = v_gt.get_image()

            v_pred = Visualizer(img, self.metadata)

            panoptic_seg, segments_info = result["panoptic_seg"]
            v_pred = v_pred.draw_panoptic_seg_predictions(panoptic_seg.cpu(), segments_info)
            pred_img = v_pred.get_image()

            vis_img = np.concatenate((anno_img, pred_img), axis=1)

            basename = os.path.basename(input["file_name"])
            if self.training:
                cv2.imwrite(
                    os.path.join(
                        self.output_dir, "training", str(storage.iter) + "_pan_" + basename
                    ),
                    vis_img[:, :, ::-1],
                )
            else:
                cv2.imwrite(
                    os.path.join(self.output_dir, "inference", "pan_" + basename),
                    vis_img[:, :, ::-1],
                )


class NMSPostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, select_box_nums_for_evaluation):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        out_mask = outputs["pred_masks"]
        bs, n_queries, n_cls = out_logits.shape
        print("PostProcessSegm", out_logits.size(), out_bbox.size(), out_mask.size())

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        all_scores = prob.view(bs, n_queries * n_cls).to(out_logits.device)
        all_indexes = torch.arange(n_queries * n_cls)[None].repeat(bs, 1).to(out_logits.device)
        all_boxes = torch.div(all_indexes, out_logits.shape[2], rounding_mode="trunc")
        all_labels = all_indexes % out_logits.shape[2]

        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, all_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        keep_inds_all = []
        for b in range(bs):
            box = boxes[b]
            score = all_scores[b]
            lbls = all_labels[b]
            mask = out_mask[b]

            pre_topk = score.topk(10000).indices
            box = box[pre_topk]
            score = score[pre_topk]
            lbls = lbls[pre_topk]

            keep_inds = batched_nms(box, score, lbls, 0.7)[:select_box_nums_for_evaluation]

            result = Instances(target_sizes[b])
            result.pred_boxes = Boxes(box[keep_inds])
            result.scores = score[keep_inds]
            result.pred_classes = lbls[keep_inds]
            results.append(result)

            keep_inds_all.append(keep_inds)

        return results, keep_inds_all


def is_thing_stuff_overlap(metadata):
    thing_classes = metadata.get("thing_classes", [])
    stuff_classes = metadata.get("stuff_classes", [])
    if len(thing_classes) == 0 or len(stuff_classes) == 0:
        return False

    if set(thing_classes).issubset(set(stuff_classes)) or set(stuff_classes).issubset(
        set(thing_classes)
    ):
        return True
    else:
        return False
