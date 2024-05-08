import contextlib
import io
import logging
import os

import pycocotools.mask as mask_util

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from fvcore.common.timer import Timer

from .odinw_categories import ODINW_CATEGORIES
from .odinw_prompts import ODINW_PROMPTS

logger = logging.getLogger(__name__)


def load_coco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        # cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        # thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        # meta.thing_classes = thing_classes

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

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        # record["height"] = img_dict["height"]
        # record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

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


def register_coco_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


_PREDEFINED_SPLITS_ODINW = {
    "odinw_AerialMaritimeDrone_large": {
        "odinw_AerialMaritimeDrone_large_train": (
            "odinw/AerialMaritimeDrone/large/train/",
            "odinw/AerialMaritimeDrone/large/train/annotations_without_background_converted.json",
        ),
        "odinw_AerialMaritimeDrone_large_val": (
            "odinw/AerialMaritimeDrone/large/valid/",
            "odinw/AerialMaritimeDrone/large/valid/annotations_without_background_converted.json",
        ),
        "odinw_AerialMaritimeDrone_large_test": (
            "odinw/AerialMaritimeDrone/large/test/",
            "odinw/AerialMaritimeDrone/large/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_AerialMaritimeDrone_tiled": {
        "odinw_AerialMaritimeDrone_tiled_train": (
            "odinw/AerialMaritimeDrone/tiled/train/",
            "odinw/AerialMaritimeDrone/tiled/train/annotations_without_background_converted.json",
        ),
        "odinw_AerialMaritimeDrone_tiled_val": (
            "odinw/AerialMaritimeDrone/tiled/valid/",
            "odinw/AerialMaritimeDrone/tiled/valid/annotations_without_background_converted.json",
        ),
        "odinw_AerialMaritimeDrone_tiled_test": (
            "odinw/AerialMaritimeDrone/tiled/test/",
            "odinw/AerialMaritimeDrone/tiled/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_AmericanSignLanguageLetters_American_Sign_Language_Letters.v1-v1.coco": {
        "odinw_AmericanSignLanguageLetters_American_Sign_Language_Letters.v1-v1.coco_train": (
            "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/",
            "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/train/annotations_without_background_converted.json",
        ),
        "odinw_AmericanSignLanguageLetters_American_Sign_Language_Letters.v1-v1.coco_val": (
            "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/valid/",
            "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/valid/annotations_without_background_converted.json",
        ),
        "odinw_AmericanSignLanguageLetters_American_Sign_Language_Letters.v1-v1.coco_test": (
            "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/test/",
            "odinw/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_Aquarium_Aquarium_Combined.v2-raw-1024.coco": {
        "odinw_Aquarium_Aquarium_Combined.v2-raw-1024.coco_train": (
            "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/",
            "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/train/annotations_without_background_converted.json",
        ),
        "odinw_Aquarium_Aquarium_Combined.v2-raw-1024.coco_val": (
            "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/",
            "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/annotations_without_background_converted.json",
        ),
        "odinw_Aquarium_Aquarium_Combined.v2-raw-1024.coco_test": (
            "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/test/",
            "odinw/Aquarium/Aquarium Combined.v2-raw-1024.coco/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_BCCD_BCCD.v3-raw.coco": {
        "odinw_BCCD_BCCD.v3-raw.coco_train": (
            "odinw/BCCD/BCCD.v3-raw.coco/train/",
            "odinw/BCCD/BCCD.v3-raw.coco/train/annotations_without_background_converted.json",
        ),
        "odinw_BCCD_BCCD.v3-raw.coco_val": (
            "odinw/BCCD/BCCD.v3-raw.coco/valid/",
            "odinw/BCCD/BCCD.v3-raw.coco/valid/annotations_without_background_converted.json",
        ),
        "odinw_BCCD_BCCD.v3-raw.coco_test": (
            "odinw/BCCD/BCCD.v3-raw.coco/test/",
            "odinw/BCCD/BCCD.v3-raw.coco/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_boggleBoards_416x416AutoOrient_export_": {
        "odinw_boggleBoards_416x416AutoOrient_export_train": (
            "odinw/boggleBoards/416x416AutoOrient/export/",
            "odinw/boggleBoards/416x416AutoOrient/export/train_annotations_without_background_converted.json",
        ),
        "odinw_boggleBoards_416x416AutoOrient_export_val": (
            "odinw/boggleBoards/416x416AutoOrient/export/",
            "odinw/boggleBoards/416x416AutoOrient/export/val_annotations_without_background_converted.json",
        ),
        "odinw_boggleBoards_416x416AutoOrient_export_test": (
            "odinw/boggleBoards/416x416AutoOrient/export/",
            "odinw/boggleBoards/416x416AutoOrient/export/test_annotations_without_background_converted.json",
        ),
    },
    "odinw_brackishUnderwater_960x540": {
        "odinw_brackishUnderwater_960x540_train": (
            "odinw/brackishUnderwater/960x540/train/",
            "odinw/brackishUnderwater/960x540/train/annotations_without_background_converted.json",
        ),
        "odinw_brackishUnderwater_960x540_val": (
            "odinw/brackishUnderwater/960x540/valid/",
            "odinw/brackishUnderwater/960x540/valid/annotations_without_background_converted.json",
        ),
        "odinw_brackishUnderwater_960x540_minival": (
            "odinw/brackishUnderwater/960x540/mini_val/",
            "odinw/brackishUnderwater/960x540/mini_val/annotations_without_background_converted.json",
        ),
        "odinw_brackishUnderwater_960x540_test": (
            "odinw/brackishUnderwater/960x540/test/",
            "odinw/brackishUnderwater/960x540/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_ChessPieces_Chess_Pieces.v23-raw.coco": {
        "odinw_ChessPieces_Chess_Pieces.v23-raw.coco_train": (
            "odinw/ChessPieces/Chess Pieces.v23-raw.coco/train/",
            "odinw/ChessPieces/Chess Pieces.v23-raw.coco/train/annotations_without_background_converted.json",
        ),
        "odinw_ChessPieces_Chess_Pieces.v23-raw.coco_val": (
            "odinw/ChessPieces/Chess Pieces.v23-raw.coco/valid/",
            "odinw/ChessPieces/Chess Pieces.v23-raw.coco/valid/annotations_without_background_converted.json",
        ),
        "odinw_ChessPieces_Chess_Pieces.v23-raw.coco_test": (
            "odinw/ChessPieces/Chess Pieces.v23-raw.coco/test/",
            "odinw/ChessPieces/Chess Pieces.v23-raw.coco/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_CottontailRabbits": {
        "odinw_CottontailRabbits_train": (
            "odinw/CottontailRabbits/train/",
            "odinw/CottontailRabbits/train/annotations_without_background_converted.json",
        ),
        "odinw_CottontailRabbits_val": (
            "odinw/CottontailRabbits/valid/",
            "odinw/CottontailRabbits/valid/annotations_without_background_converted.json",
        ),
        "odinw_CottontailRabbits_test": (
            "odinw/CottontailRabbits/test/",
            "odinw/CottontailRabbits/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_dice_mediumColor_export": {
        "odinw_dice_mediumColor_export_train": (
            "odinw/dice/mediumColor/export/",
            "odinw/dice/mediumColor/export/train_annotations_without_background_converted.json",
        ),
        "odinw_dice_mediumColor_export_val": (
            "odinw/dice/mediumColor/export/",
            "odinw/dice/mediumColor/export/val_annotations_without_background_converted.json",
        ),
        "odinw_dice_mediumColor_export_test": (
            "odinw/dice/mediumColor/export/",
            "odinw/dice/mediumColor/export/test_annotations_without_background_converted.json",
        ),
    },
    "odinw_DroneControl_Drone_Control.v3-raw.coco": {
        "odinw_DroneControl_Drone_Control.v3-raw.coco_train": (
            "odinw/DroneControl/Drone Control.v3-raw.coco/train/",
            "odinw/DroneControl/Drone Control.v3-raw.coco/train/annotations_without_background_converted.json",
        ),
        "odinw_DroneControl_Drone_Control.v3-raw.coco_val": (
            "odinw/DroneControl/Drone Control.v3-raw.coco/valid/",
            "odinw/DroneControl/Drone Control.v3-raw.coco/valid/annotations_without_background_converted.json",
        ),
        "odinw_DroneControl_Drone_Control.v3-raw.coco_minival": (
            "odinw/DroneControl/Drone Control.v3-raw.coco/mini_val/",
            "odinw/DroneControl/Drone Control.v3-raw.coco/mini_val/annotations_without_background_converted.json",
        ),
        "odinw_DroneControl_Drone_Control.v3-raw.coco_test": (
            "odinw/DroneControl/Drone Control.v3-raw.coco/test/",
            "odinw/DroneControl/Drone Control.v3-raw.coco/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_EgoHands-generic": {
        "odinw_EgoHands_generic_train": (
            "odinw/EgoHands/generic/train/",
            "odinw/EgoHands/generic/train/annotations_without_background_converted.json",
        ),
        "odinw_EgoHands_generic_val": (
            "odinw/EgoHands/generic/valid/",
            "odinw/EgoHands/generic/valid/annotations_without_background_converted.json",
        ),
        "odinw_EgoHands_generic_minival": (
            "odinw/EgoHands/generic/mini_val/",
            "odinw/EgoHands/generic/mini_val/annotations_without_background_converted.json",
        ),
        "odinw_EgoHands_generic_test": (
            "odinw/EgoHands/generic/test/",
            "odinw/EgoHands/generic/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_EgoHands-specific": {
        "odinw_EgoHands_specific_train": (
            "odinw/EgoHands/specific/train/",
            "odinw/EgoHands/specific/train/annotations_without_background_converted.json",
        ),
        "odinw_EgoHands_specific_val": (
            "odinw/EgoHands/specific/valid/",
            "odinw/EgoHands/specific/valid/annotations_without_background_converted.json",
        ),
        "odinw_EgoHands_specific_minival": (
            "odinw/EgoHands/specific/mini_val/",
            "odinw/EgoHands/specific/mini_val/annotations_without_background_converted.json",
        ),
        "odinw_EgoHands_specific_test": (
            "odinw/EgoHands/specific/test/",
            "odinw/EgoHands/specific/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_HardHatWorkers_raw": {
        "odinw_HardHatWorkers_raw_train": (
            "odinw/HardHatWorkers/raw/train/",
            "odinw/HardHatWorkers/raw/train/annotations_without_background_converted.json",
        ),
        "odinw_HardHatWorkers_raw_val": (
            "odinw/HardHatWorkers/raw/valid/",
            "odinw/HardHatWorkers/raw/valid/annotations_without_background_converted.json",
        ),
        "odinw_HardHatWorkers_raw_test": (
            "odinw/HardHatWorkers/raw/test/",
            "odinw/HardHatWorkers/raw/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_MaskWearing_raw": {
        "odinw_MaskWearing_raw_train": (
            "odinw/MaskWearing/raw/train/",
            "odinw/MaskWearing/raw/train/annotations_without_background_converted.json",
        ),
        "odinw_MaskWearing_raw_val": (
            "odinw/MaskWearing/raw/valid/",
            "odinw/MaskWearing/raw/valid/annotations_without_background_converted.json",
        ),
        "odinw_MaskWearing_raw_test": (
            "odinw/MaskWearing/raw/test/",
            "odinw/MaskWearing/raw/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_MountainDewCommercial": {
        "odinw_MountainDewCommercial_train": (
            "odinw/MountainDewCommercial/train/",
            "odinw/MountainDewCommercial/train/annotations_without_background_converted.json",
        ),
        "odinw_MountainDewCommercial_val": (
            "odinw/MountainDewCommercial/valid/",
            "odinw/MountainDewCommercial/valid/annotations_without_background_converted.json",
        ),
        "odinw_MountainDewCommercial_test": (
            "odinw/MountainDewCommercial/test/",
            "odinw/MountainDewCommercial/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco": {
        "odinw_NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco_train": (
            "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/",
            "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/train/annotations_without_background_converted.json",
        ),
        "odinw_NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco_val": (
            "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/",
            "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/annotations_without_background_converted.json",
        ),
        "odinw_NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco_test": (
            "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/test/",
            "odinw/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_openPoetryVision_512x512": {
        "odinw_openPoetryVision_512x512_train": (
            "odinw/openPoetryVision/512x512/train/",
            "odinw/openPoetryVision/512x512/train/annotations_without_background_converted.json",
        ),
        "odinw_openPoetryVision_512x512_val": (
            "odinw/openPoetryVision/512x512/valid/",
            "odinw/openPoetryVision/512x512/valid/annotations_without_background_converted.json",
        ),
        "odinw_openPoetryVision_512x512_minival": (
            "odinw/openPoetryVision/512x512/mini_val/",
            "odinw/openPoetryVision/512x512/mini_val/annotations_without_background_converted.json",
        ),
        "odinw_openPoetryVision_512x512_test": (
            "odinw/openPoetryVision/512x512/test/",
            "odinw/openPoetryVision/512x512/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_OxfordPets-by-breed": {
        "odinw_OxfordPets_by-breed_train": (
            "odinw/OxfordPets/by-breed/train/",
            "odinw/OxfordPets/by-breed/train/annotations_without_background_converted.json",
        ),
        "odinw_OxfordPets_by-breed_val": (
            "odinw/OxfordPets/by-breed/valid/",
            "odinw/OxfordPets/by-breed/valid/annotations_without_background_converted.json",
        ),
        "odinw_OxfordPets_by-breed_minival": (
            "odinw/OxfordPets/by-breed/mini_val/",
            "odinw/OxfordPets/by-breed/mini_val/annotations_without_background_converted.json",
        ),
        "odinw_OxfordPets_by-breed_test": (
            "odinw/OxfordPets/by-breed/test/",
            "odinw/OxfordPets/by-breed/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_OxfordPets-by-species": {
        "odinw_OxfordPets_by-species_train": (
            "odinw/OxfordPets/by-species/train/",
            "odinw/OxfordPets/by-species/train/annotations_without_background_converted.json",
        ),
        "odinw_OxfordPets_by-species_val": (
            "odinw/OxfordPets/by-species/valid/",
            "odinw/OxfordPets/by-species/valid/annotations_without_background_converted.json",
        ),
        "odinw_OxfordPets_by-species_minival": (
            "odinw/OxfordPets/by-species/mini_val/",
            "odinw/OxfordPets/by-species/mini_val/annotations_without_background_converted.json",
        ),
        "odinw_OxfordPets_by-species_test": (
            "odinw/OxfordPets/by-species/test/",
            "odinw/OxfordPets/by-species/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_Packages_Raw": {
        "odinw_Packages_Raw_train": (
            "odinw/Packages/Raw/train/",
            "odinw/Packages/Raw/train/annotations_without_background_converted.json",
        ),
        "odinw_Packages_Raw_val": (
            "odinw/Packages/Raw/valid/",
            "odinw/Packages/Raw/valid/annotations_without_background_converted.json",
        ),
        "odinw_Packages_Raw_test": (
            "odinw/Packages/Raw/test/",
            "odinw/Packages/Raw/test/annotations_without_background_converted.json",
            # "odinw/Packages/Raw/test/_annotations.coco_converted.json",
        ),
    },
    "odinw_PascalVOC": {
        "odinw_PascalVOC_train": (
            "odinw/PascalVOC/train/",
            "odinw/PascalVOC/train/annotations_without_background_converted.json",
        ),
        "odinw_PascalVOC_val": (
            "odinw/PascalVOC/valid/",
            "odinw/PascalVOC/valid/annotations_without_background_converted.json",
        ),
    },
    "odinw_pistols_export": {
        "odinw_pistols_export_train": (
            "odinw/pistols/export/",
            "odinw/pistols/export/train_annotations_without_background_converted.json",
        ),
        "odinw_pistols_export_val": (
            "odinw/pistols/export/",
            "odinw/pistols/export/val_annotations_without_background_converted.json",
        ),
        "odinw_pistols_export_test": (
            "odinw/pistols/export/",
            "odinw/pistols/export/test_annotations_without_background_converted.json",
        ),
    },
    "odinw_PKLot_640": {
        "odinw_PKLot_640_train": (
            "odinw/PKLot/640/train/",
            "odinw/PKLot/640/train/annotations_without_background_converted.json",
        ),
        "odinw_PKLot_640_val": (
            "odinw/PKLot/640/valid/",
            "odinw/PKLot/640/valid/annotations_without_background_converted.json",
        ),
        "odinw_PKLot_640_minival": (
            "odinw/PKLot/640/mini_val/",
            "odinw/PKLot/640/mini_val/annotations_without_background_converted.json",
        ),
        "odinw_PKLot_640_test": (
            "odinw/PKLot/640/test/",
            "odinw/PKLot/640/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_plantdoc_100x100": {
        "odinw_plantdoc_100x100_train": (
            "odinw/plantdoc/100x100/train/",
            "odinw/plantdoc/100x100/train/annotations_without_background_converted.json",
        ),
        "odinw_plantdoc_100x100_val": (
            "odinw/plantdoc/100x100/valid/",
            "odinw/plantdoc/100x100/valid/annotations_without_background_converted.json",
        ),
        "odinw_plantdoc_100x100_test": (
            "odinw/plantdoc/100x100/test/",
            "odinw/plantdoc/100x100/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_plantdoc_416x416": {
        "odinw_plantdoc_416x416_train": (
            "odinw/plantdoc/416x416/train/",
            "odinw/plantdoc/416x416/train/annotations_without_background_converted.json",
        ),
        "odinw_plantdoc_416x416_val": (
            "odinw/plantdoc/416x416/valid/",
            "odinw/plantdoc/416x416/valid/annotations_without_background_converted.json",
        ),
        "odinw_plantdoc_416x416_test": (
            "odinw/plantdoc/416x416/test/",
            "odinw/plantdoc/416x416/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_pothole": {
        "odinw_pothole_train": (
            "odinw/pothole/train/",
            "odinw/pothole/train/annotations_without_background_converted.json",
        ),
        "odinw_pothole_val": (
            "odinw/pothole/valid/",
            "odinw/pothole/valid/annotations_without_background_converted.json",
        ),
        "odinw_pothole_test": (
            "odinw/pothole/test/",
            "odinw/pothole/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_Raccoon_Raccoon.v2-raw.coco": {
        "odinw_Raccoon_Raccoon.v2-raw.coco_train": (
            "odinw/Raccoon/Raccoon.v2-raw.coco/train/",
            "odinw/Raccoon/Raccoon.v2-raw.coco/train/annotations_without_background_converted.json",
        ),
        "odinw_Raccoon_Raccoon.v2-raw.coco_val": (
            "odinw/Raccoon/Raccoon.v2-raw.coco/valid/",
            "odinw/Raccoon/Raccoon.v2-raw.coco/valid/annotations_without_background_converted.json",
        ),
        "odinw_Raccoon_Raccoon.v2-raw.coco_test": (
            "odinw/Raccoon/Raccoon.v2-raw.coco/test/",
            "odinw/Raccoon/Raccoon.v2-raw.coco/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_selfdrivingCar_fixedLarge_export": {
        "odinw_selfdrivingCar_fixedLarge_export_train": (
            "odinw/selfdrivingCar/fixedLarge/export/",
            "odinw/selfdrivingCar/fixedLarge/export/train_annotations_without_background_converted.json",
        ),
        "odinw_selfdrivingCar_fixedLarge_export_val": (
            "odinw/selfdrivingCar/fixedLarge/export/",
            "odinw/selfdrivingCar/fixedLarge/export/val_annotations_without_background_converted.json",
        ),
        "odinw_selfdrivingCar_fixedLarge_export_test": (
            "odinw/selfdrivingCar/fixedLarge/export/",
            "odinw/selfdrivingCar/fixedLarge/export/test_annotations_without_background_converted.json",
        ),
    },
    "odinw_ShellfishOpenImages_raw": {
        "odinw_ShellfishOpenImages_raw_train": (
            "odinw/ShellfishOpenImages/raw/train/",
            "odinw/ShellfishOpenImages/raw/train/annotations_without_background_converted.json",
        ),
        "odinw_ShellfishOpenImages_raw_val": (
            "odinw/ShellfishOpenImages/raw/valid/",
            "odinw/ShellfishOpenImages/raw/valid/annotations_without_background_converted.json",
        ),
        "odinw_ShellfishOpenImages_raw_test": (
            "odinw/ShellfishOpenImages/raw/test/",
            "odinw/ShellfishOpenImages/raw/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_ThermalCheetah": {
        "odinw_ThermalCheetah_train": (
            "odinw/ThermalCheetah/train/",
            "odinw/ThermalCheetah/train/annotations_without_background_converted.json",
        ),
        "odinw_ThermalCheetah_val": (
            "odinw/ThermalCheetah/valid/",
            "odinw/ThermalCheetah/valid/annotations_without_background_converted.json",
        ),
        "odinw_ThermalCheetah_test": (
            "odinw/ThermalCheetah/test/",
            "odinw/ThermalCheetah/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_thermalDogsAndPeople": {
        "odinw_thermalDogsAndPeople_train": (
            "odinw/thermalDogsAndPeople/train/",
            "odinw/thermalDogsAndPeople/train/annotations_without_background_converted.json",
        ),
        "odinw_thermalDogsAndPeople_val": (
            "odinw/thermalDogsAndPeople/valid/",
            "odinw/thermalDogsAndPeople/valid/annotations_without_background_converted.json",
        ),
        "odinw_thermalDogsAndPeople_test": (
            "odinw/thermalDogsAndPeople/test/",
            "odinw/thermalDogsAndPeople/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_UnoCards_raw": {
        "odinw_UnoCards_raw_train": (
            "odinw/UnoCards/raw/train/",
            "odinw/UnoCards/raw/train/annotations_without_background_converted.json",
        ),
        "odinw_UnoCards_raw_val": (
            "odinw/UnoCards/raw/valid/",
            "odinw/UnoCards/raw/valid/annotations_without_background_converted.json",
        ),
        "odinw_UnoCards_raw_minival": (
            "odinw/UnoCards/raw/mini_val/",
            "odinw/UnoCards/raw/mini_val/annotations_without_background_converted.json",
        ),
        "odinw_UnoCards_raw_test": (
            "odinw/UnoCards/raw/test/",
            "odinw/UnoCards/raw/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_VehiclesOpenImages_416x416": {
        "odinw_VehiclesOpenImages_416x416_train": (
            "odinw/VehiclesOpenImages/416x416/train/",
            "odinw/VehiclesOpenImages/416x416/train/annotations_without_background_converted.json",
        ),
        "odinw_VehiclesOpenImages_416x416_val": (
            "odinw/VehiclesOpenImages/416x416/valid/",
            "odinw/VehiclesOpenImages/416x416/valid/annotations_without_background_converted.json",
        ),
        "odinw_VehiclesOpenImages_416x416_minival": (
            "odinw/VehiclesOpenImages/416x416/mini_val/",
            "odinw/VehiclesOpenImages/416x416/mini_val/annotations_without_background_converted.json",
        ),
        "odinw_VehiclesOpenImages_416x416_test": (
            "odinw/VehiclesOpenImages/416x416/test/",
            "odinw/VehiclesOpenImages/416x416/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_websiteScreenshots": {
        "odinw_websiteScreenshots_train": (
            "odinw/websiteScreenshots/train/",
            "odinw/websiteScreenshots/train/annotations_without_background_converted.json",
        ),
        "odinw_websiteScreenshots_val": (
            "odinw/websiteScreenshots/valid/",
            "odinw/websiteScreenshots/valid/annotations_without_background_converted.json",
        ),
        "odinw_websiteScreenshots_minival": (
            "odinw/websiteScreenshots/mini_val/",
            "odinw/websiteScreenshots/mini_val/annotations_without_background_converted.json",
        ),
        "odinw_websiteScreenshots_test": (
            "odinw/websiteScreenshots/test/",
            "odinw/websiteScreenshots/test/annotations_without_background_converted.json",
        ),
    },
    "odinw_WildfireSmoke": {
        "odinw_WildfireSmoke_train": (
            "odinw/WildfireSmoke/train/",
            "odinw/WildfireSmoke/train/annotations_without_background_converted.json",
        ),
        "odinw_WildfireSmoke_val": (
            "odinw/WildfireSmoke/valid/",
            "odinw/WildfireSmoke/valid/annotations_without_background_converted.json",
        ),
        "odinw_WildfireSmoke_test": (
            "odinw/WildfireSmoke/test/",
            "odinw/WildfireSmoke/test/annotations_without_background_converted.json",
        ),
    },
}


def _get_builtin_metadata(name):
    meta = {}
    if name.split("_")[1] in ODINW_PROMPTS:
        meta["thing_classes"] = [
            ODINW_PROMPTS[name.split("_")[1]](m["name"])
            for m in ODINW_CATEGORIES[name.split("_")[1]]
        ]
    else:
        meta["thing_classes"] = [m["name"] for m in ODINW_CATEGORIES[name.split("_")[1]]]
    return meta


def register_all_odinw(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_ODINW.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


if __name__.endswith(".odinw_instance"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_odinw(_root)
