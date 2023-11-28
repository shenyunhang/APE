import logging
import os

import pycocotools.mask as mask_util

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta
from detectron2.data.datasets.lvis_v0_5_categories import LVIS_CATEGORIES as LVIS_V0_5_CATEGORIES
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES as LVIS_V1_CATEGORIES
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from fvcore.common.timer import Timer

from .lvis_v1_coco_category_image_count import LVIS_V1_COCO_CATEGORY_IMAGE_COUNT

"""
This file contains functions to parse LVIS-format annotations into dicts in the
"Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_d3_json", "register_d3_instances"]


def register_d3_instances(name, metadata, json_file, image_root, anno_root):
    """
    Register a dataset in LVIS's json annotation format for instance detection and segmentation.

    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    DatasetCatalog.register(name, lambda: load_d3_json(json_file, image_root, anno_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="d3", **metadata
    )


def load_d3_json(json_file, image_root, anno_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file in LVIS's annotation format.

    Args:
        json_file (str): full path to the LVIS json annotation file.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., "lvis_v0.5_train").
            If provided, this function will put "thing_classes" into the metadata
            associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "bbox", "bbox_mode", "category_id",
            "segmentation"). The values for these keys will be returned as-is.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from d_cube import D3

    timer = Timer()

    d3 = D3(image_root, anno_root)

    if timer.seconds() > 1:
        logger.info("Loading d3 takes {:.2f} seconds.".format(timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(d3.get_sent_ids())
        cats = d3.load_sents(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["raw_sent"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    img_ids = d3.get_img_ids()
    imgs = d3.load_imgs(img_ids)
    anno_ids = [d3.get_anno_ids(img_ids=img_id) for img_id in img_ids]
    anns = [d3.load_annos(anno_ids=anno_id) for anno_id in anno_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(d3.load_annos())
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{anno_root} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "sent_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        if meta.group == "intra":
            group_ids = d3.get_group_ids(img_ids=[image_id])
            sent_ids = d3.get_sent_ids(group_ids=group_ids)
            sent_list = d3.load_sents(sent_ids=sent_ids)
            # assert len(anno_dict_list) == len(sent_ids)
        elif meta.group == "inter":
            sent_ids = d3.get_sent_ids()
            sent_list = d3.load_sents(sent_ids=sent_ids)
            # sent_list = d3.load_sents()
        else:
            assert False
        ref_list = [sent["raw_sent"] for sent in sent_list]
        record["expressions"] = ref_list
        if id_map:
            record["sent_ids"] = [id_map[x] for x in sent_ids]

        objs = []
        for anno in anno_dict_list:
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

            assert len(obj["bbox"]) == 1
            obj["bbox"] = list(obj["bbox"][0])
            # assert len(obj["sent_id"]) == 1
            obj["sent_id"] = obj["sent_id"][0]

            segm = anno.get("segmentation", None)
            assert len(segm) == 1
            segm = segm[0]
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

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["sent_id"]
                try:
                    obj["sent_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered sent_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            obj["category_id"] = obj["sent_id"]
            obj["iscrowd"] = 0
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts


def get_d3_instances_meta(dataset_name):
    if "intra_scenario" in dataset_name:
        group = "intra"
    elif "inter_scenario" in dataset_name:
        group = "inter"
    else:
        assert False
    return {"group": group}


_PREDEFINED_SPLITS_D3 = {
    "d3_inter_scenario": {
        "d3_inter_scenario": (
            "D3/d3_images/",
            {
                "FULL": "D3/d3_json/d3_full_annotations.json",
                "PRES": "D3/d3_json/d3_pres_annotations.json",
                "ABS": "D3/d3_json/d3_abs_annotations.json",
            },
            "D3/d3_pkl/",
        ),
    },
    "d3_intra_scenario": {
        "d3_intra_scenario": (
            "D3/d3_images/",
            {
                "FULL": "D3/d3_json/d3_full_annotations.json",
                "PRES": "D3/d3_json/d3_pres_annotations.json",
                "ABS": "D3/d3_json/d3_abs_annotations.json",
            },
            "D3/d3_pkl/",
        ),
    },
}


def register_all_D3(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_D3.items():
        for key, (image_root, json_file, anno_root) in splits_per_dataset.items():
            register_d3_instances(
                key,
                get_d3_instances_meta(dataset_name),
                # os.path.join(root, json_file) if "://" not in json_file else json_file,
                {k: os.path.join(root, v) for k, v in json_file.items()},
                os.path.join(root, image_root),
                os.path.join(root, anno_root),
            )


if __name__.endswith(".d_cube"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_D3(_root)
