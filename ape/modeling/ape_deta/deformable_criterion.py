import copy
import logging
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from detrex.layers import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from detrex.modeling import SetCriterion
from detrex.modeling.criterion.criterion import sigmoid_focal_loss
from detrex.modeling.losses import dice_loss
from detrex.utils import get_world_size, is_dist_avail_and_initialized

from .misc import nested_tensor_from_tensor_list

logger = logging.getLogger(__name__)


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class DeformableCriterion(SetCriterion):
    """This class computes the loss for Deformable-DETR
    and two-stage Deformable-DETR
    """

    def __init__(
        self,
        num_classes,
        matcher,
        matcher_stage1,
        matcher_stage2,
        weight_dict,
        losses: List[str] = ["class", "boxes"],
        eos_coef: float = 0.1,
        loss_class_type: str = "focal_loss",
        alpha: float = 0.25,
        gamma: float = 2.0,
        use_fed_loss: bool = False,
        get_fed_loss_cls_weights: Optional[Callable] = None,
        fed_loss_num_classes: int = 50,
        fed_loss_pad_type: str = None,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        train_positive_proposal_only: bool = False,
    ):
        super(DeformableCriterion, self).__init__(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            eos_coef=eos_coef,
            loss_class_type=loss_class_type,
            alpha=alpha,
            gamma=gamma,
        )

        self.matcher_stage1 = matcher_stage1
        self.matcher_stage2 = matcher_stage2

        self.use_fed_loss = use_fed_loss
        if self.use_fed_loss:
            fed_loss_cls_weights = get_fed_loss_cls_weights()
            logger.info(
                f"fed_loss_cls_weights: {fed_loss_cls_weights.size()} num_classes: {num_classes}"
            )

            if len(fed_loss_cls_weights) < num_classes:
                if fed_loss_pad_type == "max":
                    fed_loss_pad_value = fed_loss_cls_weights.max().item()
                elif fed_loss_pad_type == "max1000":
                    fed_loss_pad_value = fed_loss_cls_weights.max().item() * 1000
                elif fed_loss_pad_type == "mean":
                    fed_loss_pad_value = fed_loss_cls_weights.mean().item()
                elif fed_loss_pad_type == "median":
                    fed_loss_pad_value = fed_loss_cls_weights.median().item()
                elif fed_loss_pad_type == "cat":
                    fed_loss_pad_classes = torch.arange(len(fed_loss_cls_weights), num_classes)
                    self.register_buffer("fed_loss_pad_classes", fed_loss_pad_classes)
                    fed_loss_pad_value = 0
                else:
                    fed_loss_pad_value = torch.kthvalue(
                        fed_loss_cls_weights, int(num_classes * 7.0 / 10)
                    )[0].item()

                logger.info(
                    f"pad fed_loss_cls_weights with type {fed_loss_pad_type} and value {fed_loss_pad_value}"
                )
                if getattr(self, "fed_loss_pad_classes", None) is not None:
                    logger.info(f"pad fed_loss_classes with {self.fed_loss_pad_classes}")
                fed_loss_cls_weights = torch.cat(
                    (
                        fed_loss_cls_weights,
                        fed_loss_cls_weights.new_full(
                            (num_classes - len(fed_loss_cls_weights),),
                            fed_loss_pad_value,
                        ),
                    ),
                    dim=0,
                )

                logger.info(f"fed_loss_cls_weights: {fed_loss_cls_weights[-100:]}")
                logger.info(
                    f"fed_loss_cls_weights: {fed_loss_cls_weights.size()} num_classes: {num_classes}"
                )

            assert (
                len(fed_loss_cls_weights) == self.num_classes
            ), "Please check the provided fed_loss_cls_weights. Their size should match num_classes"
            self.register_buffer("fed_loss_cls_weights", fed_loss_cls_weights)
        self.fed_loss_num_classes = fed_loss_num_classes

        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.train_positive_proposal_only = train_positive_proposal_only
        self.alpha_old = self.alpha

    def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes, num_classes, weight):
        """
        Args:
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
            num_fed_loss_classes: minimum number of classes to keep when calculating federated loss.
            Will sample negative classes if number of unique gt_classes is smaller than this value.
            num_classes: number of foreground classes
            weight: probabilities used to sample negative classes

        Returns:
            Tensor:
                classes to keep when calculating the federated loss, including both unique gt
                classes and sampled negative classes.
        """
        unique_gt_classes = torch.unique(gt_classes)
        prob = unique_gt_classes.new_ones(num_classes + 1).float()
        prob[-1] = 0
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.float().clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
        else:
            fed_loss_classes = unique_gt_classes
        return fed_loss_classes

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        if self.loss_class_type == "ce_loss":
            num_classes = src_logits.shape[2] - 1
        elif self.loss_class_type == "focal_loss":
            num_classes = src_logits.shape[2]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        if self.loss_class_type == "ce_loss":
            loss_class = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )
        elif (
            self.loss_class_type == "focal_loss"
            and self.use_fed_loss
            and num_classes == len(self.fed_loss_cls_weights)
        ):
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            fed_loss_classes = self.get_fed_loss_classes(
                target_classes_o,
                num_fed_loss_classes=self.fed_loss_num_classes,
                num_classes=target_classes_onehot.shape[2],
                weight=self.fed_loss_cls_weights,
            )

            if getattr(self, "fed_loss_pad_classes", None) is not None:
                fed_loss_classes = torch.cat([fed_loss_classes, self.fed_loss_pad_classes])
                fed_loss_classes = torch.unique(fed_loss_classes)

            loss_class = (
                sigmoid_focal_loss(
                    src_logits[:, :, fed_loss_classes],
                    target_classes_onehot[:, :, fed_loss_classes],
                    num_boxes=num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
                * src_logits.shape[1]
            )
        elif self.loss_class_type == "focal_loss":
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_class = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes=num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
                * src_logits.shape[1]
            )

        if not torch.isfinite(loss_class):
            print("loss_class", loss_class)
            print("outputs", outputs)
            print("targets", targets)
            print("indices", indices)
            print("num_boxes", num_boxes)

        losses = {"loss_class": loss_class}

        return losses

    def loss_anchor_ious(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        ious = torch.cat([t["ious"][J, I] for t, (I, J) in zip(targets, indices)])
        predictions = torch.cat([p[I] for p, (I, _) in zip(src_logits, indices)])

        predictions = predictions.squeeze(1)

        loss_iou = F.mse_loss(predictions, ious, size_average=None, reduce=None, reduction="mean")

        losses = {"loss_iou": loss_iou}

        return losses

    def loss_pred_ious(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        iou, _ = box_iou(
            box_cxcywh_to_xyxy(target_boxes),
            box_cxcywh_to_xyxy(src_boxes),
        )
        ious = iou[range(len(iou)), range(len(iou))]

        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"][idx]
        src_logits = src_logits.squeeze(1)

        loss_iou = F.mse_loss(src_logits, ious, size_average=None, reduce=None, reduction="mean")

        losses = {"loss_iou": loss_iou}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def loss_boxes_panoptic(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if "is_thing" in targets[0]:
            is_thing = torch.cat([t["is_thing"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            if is_thing.sum() == 0:  # no gt
                losses = {}
                losses["loss_bbox"] = src_boxes.sum() * 0.0
                losses["loss_giou"] = src_boxes.sum() * 0.0
                return losses
            target_boxes = target_boxes[is_thing]
            src_boxes = src_boxes[is_thing]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        if outputs["pred_masks"] is None:
            return {}
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        max_mask_num = 128 * len(indices)
        if src_idx[0].size(0) > max_mask_num:
            perm = torch.sort(torch.randperm(src_idx[0].size(0))[:max_mask_num])[0]

            src_idx = (src_idx[0][perm], src_idx[1][perm])
            tgt_idx = (tgt_idx[0][perm], tgt_idx[1][perm])

        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()

        if target_masks.size(1) == 0:  # no gt
            losses = {}
            losses["loss_mask"] = src_masks.sum() * 0.0
            losses["loss_dice"] = src_masks.sum() * 0.0
            return losses

        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        src_masks = F.interpolate(
            src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(
                src_masks.sigmoid(), target_masks, reduction="mean", avg_factor=num_boxes
            ),
        }
        del src_masks
        del target_masks
        return losses

    def loss_masks_maskdino(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        if outputs["pred_masks"] is None:
            return {}
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        if not isinstance(src_masks, torch.Tensor):
            mask_embeds = src_masks["mask_embeds"]
            mask_features = src_masks["mask_features"]
            src_masks = torch.cat(
                [
                    torch.einsum("qc,chw->qhw", mask_embeds[i][src], mask_features[i])
                    for i, (src, _) in enumerate(indices)
                ],
                dim=0,
            )
        else:
            src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()

        if target_masks.size(1) == 0:  # no gt
            losses = {}
            losses["loss_mask_maskdino"] = src_masks.sum() * 0.0
            losses["loss_dice_maskdino"] = src_masks.sum() * 0.0
            return losses

        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask_maskdino": sigmoid_ce_loss(point_logits, point_labels, num_boxes),
            "loss_dice_maskdino": dice_loss(
                point_logits.sigmoid(), point_labels, reduction="mean", avg_factor=num_boxes
            ),
        }

        del src_masks
        del target_masks
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "class": self.loss_labels,
            "boxes": self.loss_boxes,
            "boxes_panoptic": self.loss_boxes_panoptic,
            "masks": self.loss_masks,
            "masks_maskdino": self.loss_masks_maskdino,
            "anchor_iou": self.loss_anchor_ious,
            "pred_iou": self.loss_pred_ious,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"
        }

        if self.matcher_stage2 is not None:
            indices = self.matcher_stage2(outputs_without_aux, targets)
        else:
            indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        if "is_thing" in targets[0] and False:
            unique_classes = torch.cat([t["labels"] for t in targets], dim=0)
            is_thing = torch.cat([t["is_thing"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            all_classes = torch.cat([t["labels"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            thing_classes = all_classes[is_thing]
            stuff_classes = all_classes[~is_thing]

            print(
                "thing_classes",
                1.0 * len(thing_classes) / max(len(torch.unique(thing_classes)), 1),
                "stuff_classes",
                1.0 * len(stuff_classes) / max(len(torch.unique(stuff_classes)), 1),
            )

        losses = {}
        for loss in self.losses:
            if loss == "pred_iou" or loss == "anchor_iou":
                continue
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if self.matcher_stage2 is not None:
                    pass
                else:
                    indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        continue
                    if loss == "pred_iou" or loss == "anchor_iou":
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            if self.train_positive_proposal_only:
                self.alpha = 1.0
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
                if "is_thing" in bt:
                    del bt["is_thing"]
            if self.matcher_stage1 is not None:
                indices, ious = self.matcher_stage1(
                    enc_outputs, bin_targets, return_cost_matrix=True
                )
                for bt, iou in zip(bin_targets, ious):
                    bt["ious"] = iou
            else:
                indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == "masks":
                    continue
                if loss == "masks_maskdino":
                    continue
                if loss == "class" and ("pred_iou" in losses or "anchor_iou" in losses):
                    continue
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)
            if self.train_positive_proposal_only:
                self.alpha = self.alpha_old

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "matcher_stage1: {}".format(self.matcher_stage1),
            "matcher_stage2: {}".format(self.matcher_stage2),
            "losses: {}".format(self.losses),
            "loss_class_type: {}".format(self.loss_class_type),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "focal loss alpha: {}".format(self.alpha),
            "focal loss gamma: {}".format(self.gamma),
            "use_fed_loss: {}".format(self.use_fed_loss),
            "fed_loss_num_classes: {}".format(self.fed_loss_num_classes),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
