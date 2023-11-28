import warnings
from typing import List, Tuple

import torch

from detectron2.layers import batched_nms
from detectron2.structures import Boxes, Instances

__all__ = [
    "fast_rcnn_inference",
]


"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    use_soft_nms: bool = False,
    soft_nms_method: str = "linear",
    soft_nms_iou_threshold: float = 0.3,
    soft_nms_sigma: float = 0.5,
    soft_nms_class_wise: bool = False,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image,
            scores_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image,
            use_soft_nms=use_soft_nms,
            soft_nms_method=soft_nms_method,
            soft_nms_iou_threshold=soft_nms_iou_threshold,
            soft_nms_sigma=soft_nms_sigma,
            soft_nms_class_wise=soft_nms_class_wise,
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    use_soft_nms: bool = False,
    soft_nms_method: str = "linear",
    soft_nms_iou_threshold: float = 0.3,
    soft_nms_sigma: float = 0.5,
    soft_nms_class_wise: bool = False,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    filter_mask = scores > score_thresh  # R x K
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    if use_soft_nms:
        from mmcv.ops import soft_nms

        if not soft_nms_class_wise:
            dets, keep = soft_nms(
                boxes=boxes,
                scores=scores,
                iou_threshold=soft_nms_iou_threshold,
                sigma=soft_nms_sigma,
                min_score=1e-3,
                method=soft_nms_method,
            )
            boxes, scores = dets[:, :4], dets[:, -1]
        else:
            try:
                max_coordinate = boxes.max()
            except:
                print(boxes.shape)  # empty
                warnings.warn("setting max_coordinate to 0")
                max_coordinate = 0
            idxs = filter_inds[:, 1]
            offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]
            dets, keep = soft_nms(
                boxes=boxes_for_nms,
                scores=scores,
                iou_threshold=soft_nms_iou_threshold,
                sigma=soft_nms_sigma,
                min_score=1e-3,
                method=soft_nms_method,
            )

            if topk_per_image >= 0:
                keep = keep[:topk_per_image]
            boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

            result = Instances(image_shape)
            result.pred_boxes = Boxes(boxes)
            result.scores = scores
            result.pred_classes = filter_inds[:, 1]
            return result, filter_inds[:, 0]

            boxes = boxes[keep]
            scores = dets[:, -1]  # scores are updated in soft-nms

        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        filter_inds = filter_inds[keep]
        result.pred_classes = filter_inds[:, 1]
        return result, filter_inds[:, 0]

    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]
