# -*- coding: utf-8 -*-
import copy
import json
import logging
import os
import random
import re

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from scipy.ndimage import gaussian_filter

from detectron2.data import detection_utils as utils
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    PolygonMasks,
    polygons_to_bitmask,
)
from fvcore.transforms.transform import HFlipTransform

__all__ = [
    "copypaste",
    "maybe_load_annotation_from_file",
]


def clean_string(phrase):
    # return re.sub(r"([.,'!?\"()*#:;])", "", phrase.lower()).replace("-", " ").replace("/", " ")

    phrase = re.sub(r"([.,'!?\"()*#:;])", "", phrase.lower()).replace("-", " ").replace("/", " ")
    phrase = phrase.strip("\n").strip("\r").strip().lstrip(" ").rstrip(" ")
    phrase = re.sub(" +", " ", phrase)

    replacements = {
        "½": "half",
        "—": "-",
        "™": "",
        "¢": "cent",
        "ç": "c",
        "û": "u",
        "é": "e",
        "°": " degree",
        "è": "e",
        "…": "",
    }
    for k, v in replacements.items():
        phrase = phrase.replace(k, v)

    return phrase


def transform_phrases(phrases, transforms):
    # clean
    phrases = [clean_string(phrase) for phrase in phrases]
    # hflip
    for x in transforms:
        if isinstance(x, HFlipTransform):
            phrases = [
                phrase.replace("left", "@").replace("right", "left").replace("@", "right")
                for phrase in phrases
            ]
    return phrases


def transform_expressions(expressions, transforms):
    # pick one expression if there are multiple expressions
    expression = expressions[np.random.choice(len(expressions))]
    expression = clean_string(expression)
    # deal with hflip for expression
    for x in transforms:
        if isinstance(x, HFlipTransform):
            expression = (
                expression.replace("left", "@").replace("right", "left").replace("@", "right")
            )
    return expression


def has_ordinal_num(phrases):
    # oridinal numbers
    ordinal_nums = [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
    ]

    flag = False
    for phrase in phrases:
        phrase_low = phrase.lower()
        for word in ordinal_nums:
            if word in phrase_low:
                flag = True
                break
        if flag == True:
            break
    return flag


# from detectron2/utils/visualizer.py
def mask_to_polygons_2(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    res = [x + 0.5 for x in res if len(x) >= 6]
    return res, has_holes


# from detectron2/utils/visualizer.py
def mask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    res = [x + 0.5 for x in res if len(x) >= 6]
    return res, has_holes


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


# from pycococreatortools/pycococreatortools.py
def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """ ""
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode="constant", constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def instances_to_annotations(instances, img_id, bbox_mode, instance_mask_format):
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.gt_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, bbox_mode)
    boxes = boxes.tolist()
    classes = instances.gt_classes.tolist()

    if instance_mask_format == "polygon":
        segms = [[p.reshape(-1) for p in mask] for mask in instances.gt_masks]

    elif instance_mask_format == "bitmask" and False:
        masks = [np.array(mask, dtype=np.uint8) for mask in instances.gt_masks]

    else:
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.gt_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    annotations = []
    for k in range(num_instance):
        anno = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "bbox_mode": bbox_mode,
        }
        if instance_mask_format == "polygon":
            anno["segmentation"] = segms[k]
        elif instance_mask_format == "bitmask" and False:
            anno["segmentation"] = masks[k]
        else:
            anno["segmentation"] = rles[k]
        annotations.append(anno)

    return annotations


def copypaste(dataset_dict, dataset_dict_bg, image_format, instance_mask_format):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # USER: Write your own image loading if it's not from a file
    image = utils.read_image(dataset_dict["file_name"], format=image_format)
    utils.check_image_size(dataset_dict, image)

    dataset_dict_bg = copy.deepcopy(dataset_dict_bg)  # it will be modified by code below
    # USER: Write your own image loading if it's not from a file
    image_bg = utils.read_image(dataset_dict_bg["file_name"], format=image_format)
    utils.check_image_size(dataset_dict_bg, image_bg)

    image_bg = image_bg.copy()

    image_size = image_shape = image.shape[:2]  # h, w
    image_size_bg = image_shape_bg = image_bg.shape[:2]  # h, w

    instances = utils.annotations_to_instances(
        # dataset_dict["annotations"],
        [obj for obj in dataset_dict["annotations"] if obj.get("iscrowd", 0) == 0],
        image_shape,
        mask_format=instance_mask_format,
    )
    if "annotations" in dataset_dict_bg:
        instances_bg = utils.annotations_to_instances(
            # dataset_dict_bg["annotations"],
            [obj for obj in dataset_dict_bg["annotations"] if obj.get("iscrowd", 0) == 0],
            image_shape_bg,
            mask_format=instance_mask_format,
        )
    else:
        instances_bg = None

    if instances_bg is None or len(instances_bg) == 0:
        bitmasks_bg = torch.zeros((1, image_size_bg[0], image_size_bg[1])).to(torch.bool)
    elif instance_mask_format == "polygon":
        bitmasks_bg = [
            polygons_to_bitmask(polygon, *image_size_bg) for polygon in instances_bg.gt_masks
        ]
        bitmasks_bg = torch.tensor(np.array(bitmasks_bg))
    else:
        bitmasks_bg = instances_bg.gt_masks.tensor

    if instance_mask_format == "polygon":
        bitmasks = [polygons_to_bitmask(polygon, *image_size) for polygon in instances.gt_masks]
        bitmasks = torch.tensor(np.array(bitmasks))
    else:
        bitmasks = instances.gt_masks.tensor

    assert bitmasks_bg.dtype == torch.bool, bitmasks_bg.dtype
    # foreground_mask = torch.sum(bitmasks_bg, dim=0)
    foreground_mask = torch.max(bitmasks_bg, dim=0)[0]
    copypaste_mask = torch.zeros_like(foreground_mask)

    if instance_mask_format == "polygon":
        mask_areas = instances.gt_masks.area().numpy()
    else:
        mask_areas = instances.gt_masks.tensor.sum(dim=1).sum(dim=1).numpy()

    instance_list = []
    for i in mask_areas.argsort():
        i = int(i)

        box = instances.gt_boxes[i].tensor.numpy()[0]
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        if x1 + 1 > x2 or y1 + 1 > y2:
            continue

        image_p = image[y1:y2, x1:x2, :]
        bitmasks_p = bitmasks[i, y1:y2, x1:x2]

        h, w = bitmasks_p.shape

        trial = 10
        for _ in range(trial):
            if w + 10 >= image_size_bg[1] or h + 10 >= image_size_bg[0]:
                break

            x1 = random.randint(0, image_size_bg[1] - w)
            y1 = random.randint(0, image_size_bg[0] - h)
            x2 = x1 + w
            y2 = y1 + h

            bitmask = torch.zeros_like(foreground_mask)
            bitmask[y1:y2, x1:x2] = bitmasks_p

            # bitmask = bitmask * (1 - foreground_mask)
            bitmask = bitmask & (~foreground_mask)

            if bitmask.sum() < 100:
                continue

            instance = Instances(image_size_bg)
            instance.gt_classes = instances[i].gt_classes

            # if bitmask.sum() < bitmasks_p.sum():
            bitmasks_p = bitmask[y1:y2, x1:x2]

            if instance_mask_format == "polygon":
                mask = [mask_to_polygons(bitmask)[0]]
                instance.gt_masks = PolygonMasks(mask)
            else:
                instance.gt_masks = BitMasks(bitmask.unsqueeze(0))

            bitmasks_p = bitmasks_p.numpy()
            if bitmask.sum() > 128 * 64:
                bitmasks_p = gaussian_filter(bitmasks_p.astype(float), sigma=5, truncate=1)

            image_bg_p = image_bg[y1:y2, x1:x2, :]
            image_fgbg_p = image_p * bitmasks_p[..., np.newaxis] + image_bg_p * (
                1 - bitmasks_p[..., np.newaxis]
            )

            image_bg[y1:y2, x1:x2, :] = image_fgbg_p

            foreground_mask = foreground_mask | bitmask
            copypaste_mask = copypaste_mask | bitmask

            instance_list.append(instance)
            break

    if len(instance_list) > 0:
        instances = Instances.cat(instance_list)
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

        image_id = dataset_dict["image_id"]
        bbox_mode = dataset_dict["annotations"][0]["bbox_mode"]
        annotations = instances_to_annotations(instances, image_id, bbox_mode, instance_mask_format)

        for annotation in annotations:
            annotation["copypaste"] = 1

        if "annotations" in dataset_dict_bg:
            dataset_dict_bg["annotations"] += annotations
        else:
            dataset_dict_bg["annotations"] = annotations

        dataset_dict_bg["image_id"] = (
            str(dataset_dict["image_id"]) + "_" + str(dataset_dict_bg["image_id"])
        )

        dataset_dict_bg["copypaste_mask"] = copypaste_mask.numpy()

        return image_bg, dataset_dict_bg
    else:
        return None, None


def maybe_load_annotation_from_file(record, meta=None, extra_annotation_keys=None):

    file_name = record["file_name"]
    image_ext = file_name.split(".")[-1]
    file_name = file_name[: -len(image_ext)] + "json"

    if not os.path.isfile(file_name):
        return record

    try:
        with open(file_name, "r") as f:
            json_data = json.load(f)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"json.load fails: {file_name}")
        logger.warning(f"json.load fails: {e}")
        return record
    if "image" not in json_data or "annotations" not in json_data:
        return record

    image_id = record["image_id"]
    if "image_id" in json_data["image"]:
        assert json_data["image"]["image_id"] == image_id
    if "id" in json_data["image"]:
        assert json_data["image"]["id"] == image_id

    id_map = meta.thing_dataset_id_to_contiguous_id if meta is not None else None
    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    ann_keys += ["phrase", "isobject"]

    num_instances_without_valid_segmentation = 0

    if True:
        anno_dict_list = json_data["annotations"]

        objs = []
        for anno in anno_dict_list:
            if "image_id" not in anno:
                anno["image_id"] = image_id
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            # phrase = anno.get("phrase", None)
            # if phrase:
            #     obj["phrase"] = phrase

            # isobject = anno.get("isobject", None)
            # if isobject:
            #     obj["isobject"] = isobject

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs

    return record
