import copy
import logging
import os
import random
from typing import List, Optional, Union

import cv2
import numpy as np
import torch

import detectron2.utils.comm as comm
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper as DatasetMapper_d2
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import BitMasks, Boxes, Instances

from . import detection_utils as utils_ape
from . import mapper_utils

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper_copypaste"]


class DatasetMapper_copypaste(DatasetMapper_d2):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        augmentations_d2: List[Union[T.Augmentation, T.Transform]],
        augmentations_aa: List[Union[T.Augmentation, T.Transform]],
        augmentations_lsj: List[Union[T.Augmentation, T.Transform]],
        augmentations_type: List[str],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        copypaste_prob: float = 0.5,
        output_dir: str = None,
        vis_period: int = 0,
        dataset_names: tuple = (),
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.augmentations_d2       = T.AugmentationList(augmentations_d2)
        self.augmentations_aa       = T.AugmentationList(augmentations_aa)
        self.augmentations_lsj      = T.AugmentationList(augmentations_lsj)
        self.augmentations_type     = augmentations_type
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
        logger.info(f"[DatasetMapper] D2 Augmentations D2 used in {mode}: {augmentations_d2}")
        logger.info(f"[DatasetMapper] AA Augmentations used in {mode}: {augmentations_aa}")
        logger.info(f"[DatasetMapper] LSJ Augmentations used in {mode}: {augmentations_lsj}")
        logger.info(f"[DatasetMapper] Type Augmentations used in {mode}: {augmentations_type}")

        if output_dir is not None:
            self.output_dir = os.path.join(output_dir, "vis_mapper")
            os.makedirs(self.output_dir, exist_ok=True)

        self.copypaste_prob = copypaste_prob
        self.vis_period = vis_period
        self.iter = 0
        self.dataset_names = dataset_names

        self.metatada_list = []
        for dataset_name in self.dataset_names:
            metadata = MetadataCatalog.get(dataset_name)
            self.metatada_list.append(metadata)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils_ape.build_augmentation(cfg, is_train)
        augs_d2 = utils.build_augmentation(cfg, is_train)
        augs_aa = utils_ape.build_augmentation_aa(cfg, is_train)
        augs_lsj = utils_ape.build_augmentation_lsj(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            raise NotImplementedError("cfg.INPUT.CROP.ENABLED is not supported yet")
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        if cfg.INPUT.MASK_FORMAT == "polygon":
            logger = logging.getLogger(__name__)
            logger.warning("Using polygon is slow, use bitmask instead")
        if cfg.INPUT.MASK_FORMAT == "bitmask":
            logger = logging.getLogger(__name__)
            logger.warning("Using bitmask may has bug, use polygon instead")
            assert (
                cfg.INPUT.SEG_PAD_VALUE == 0
            ), "PadTransform should pad bitmask with value 0. Please setting cfg.INPUT.SEG_PAD_VALUE to 0. \nNoted that cfg.INPUT.SEG_PAD_VALUE is also used to pad semantic segmentation. If semantic segmentation is used, Please set cfg.INPUT.FORMAT to polygon."

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "augmentations_d2": augs_d2,
            "augmentations_aa": augs_aa,
            "augmentations_lsj": augs_lsj,
            "augmentations_type": cfg.INPUT.AUGMENT_TYPE,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "output_dir": cfg.OUTPUT_DIR,
            "copypaste_prob": cfg.DATASETS.COPYPASTE.PROB,
            "vis_period": cfg.VIS_PERIOD,
            "dataset_names": cfg.DATASETS.TRAIN,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        copypaste = [
            obj.get("copypaste", 0)
            for obj in dataset_dict["annotations"]
            if obj.get("iscrowd", 0) == 0
        ]

        phrases = [
            obj.get("phrase", "")
            for obj in dataset_dict["annotations"]
            if obj.get("iscrowd", 0) == 0
        ]

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        instances.copypaste = torch.tensor(copypaste)

        if sum([len(x) for x in phrases]) > 0:
            instances.phrase_idxs = torch.tensor(range(len(phrases)))

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes and instances.has("gt_masks"):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances, box_threshold=10)

        if sum([len(x) for x in phrases]) > 0:
            phrases_filtered = []
            for x in dataset_dict["instances"].phrase_idxs.tolist():
                phrases_filtered.append(phrases[x])
            dataset_dict["instances"].phrases = mapper_utils.transform_phrases(
                phrases_filtered, transforms
            )
            dataset_dict["instances"].remove("phrase_idxs")
            # dataset_dict["instances"].gt_classes = torch.tensor(range(len(phrases_filtered)))

    def __call__(self, dataset_dict, dataset_dict_bg):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"read_image fails: {dataset_dict['file_name']}")
            logger.error(f"read_image fails: {e}")
            return None
        utils.check_image_size(dataset_dict, image)

        # ------------------------------------------------------------------------------------
        if (
            self.is_train
            and "annotations" in dataset_dict
            and (
                len(dataset_dict["annotations"]) == 0
                or any(["bbox" not in anno for anno in dataset_dict["annotations"]])
            )
        ):
            if "dataset_id" in dataset_dict:
                dataset_id = dataset_dict["dataset_id"]
            else:
                dataset_id = 0
            metadata = self.metatada_list[dataset_id]
            if "sa1b" in self.dataset_names[dataset_id]:
                metadata = None
            dataset_dict = mapper_utils.maybe_load_annotation_from_file(dataset_dict, meta=metadata)

            for anno in dataset_dict["annotations"]:
                if "bbox" not in anno:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Box not found: {dataset_dict}")
                    return None
                if "category_id" not in anno:
                    anno["category_id"] = 0
        # ------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------
        if dataset_dict["copypaste"] and self.copypaste_prob > random.uniform(0, 1):
            image_cp, dataset_dict_cp = mapper_utils.copypaste(
                dataset_dict, dataset_dict_bg, self.image_format, self.instance_mask_format
            )

            if dataset_dict_cp is None or image_cp is None:
                pass
            else:
                for key in dataset_dict.keys():
                    if key in dataset_dict_cp:
                        continue
                    dataset_dict_cp[key] = dataset_dict[key]
                dataset_dict = dataset_dict_cp
                image = image_cp
        # ------------------------------------------------------------------------------------

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            try:
                sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"read_image fails: {e}")
                logger.error(f"read_image fails: {dataset_dict}")
                return None

            if "copypaste_mask" in dataset_dict:
                # assume thing class is 0
                sem_seg_gt = sem_seg_gt.copy()
                sem_seg_gt[dataset_dict["copypaste_mask"]] = 0
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        try:
            if "dataset_id" not in dataset_dict or dataset_dict["dataset_id"] >= len(
                self.augmentations_type
            ):
                transforms = self.augmentations(aug_input)
            elif self.augmentations_type[dataset_dict["dataset_id"]] == "D2":
                transforms = self.augmentations_d2(aug_input)
            elif self.augmentations_type[dataset_dict["dataset_id"]] == "AA":
                transforms = self.augmentations_aa(aug_input)
            elif self.augmentations_type[dataset_dict["dataset_id"]] == "LSJ":
                transforms = self.augmentations_lsj(aug_input)
            else:
                print("fall back to default augmentation")
                transforms = self.augmentations(aug_input)
            image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"augment fails: {dataset_dict['file_name']}")
            logger.error(f"augment fails: {e}")
            return None

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        # seperate box and region
        if "annotations" in dataset_dict:
            annotations = []
            annotations_phrase = []
            for ann in dataset_dict.pop("annotations"):
                if ann.get("isobject", 1) == 0:
                    annotations_phrase.append(ann)
                else:
                    annotations.append(ann)
            if len(annotations_phrase) > 0:
                dataset_dict["annotations"] = annotations_phrase
                self._transform_annotations(dataset_dict, transforms, image_shape)
                dataset_dict["instances_phrase"] = dataset_dict.pop("instances")
            dataset_dict["annotations"] = annotations

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        # ------------------------------------------------------------------------------------
        if self.vis_period > 0 and self.iter % self.vis_period == 0:
            self.visualize_training(dataset_dict)
        # ------------------------------------------------------------------------------------
        self.iter += 1

        return dataset_dict

    def visualize_training(self, dataset_dict, prefix="", suffix=""):
        if self.output_dir is None:
            return
        if dataset_dict is None:
            return
        # if "instances" not in dataset_dict:
        #     return
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog

        if "dataset_id" in dataset_dict:
            dataset_id = dataset_dict["dataset_id"]
        else:
            dataset_id = 0
        dataset_name = self.dataset_names[dataset_id]
        metadata = MetadataCatalog.get(dataset_name)
        class_names = metadata.get(
            "thing_classes",
            [
                "thing",
            ],
        )

        img = dataset_dict["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.image_format)
        image_shape = img.shape[:2]  # h, w
        vis = Visualizer(img, metadata=metadata)
        if "instances" in dataset_dict:
            vis = vis.overlay_instances(
                boxes=dataset_dict["instances"].gt_boxes,
                masks=dataset_dict["instances"].gt_masks
                if dataset_dict["instances"].has("gt_masks")
                else None,
                labels=[class_names[i] for i in dataset_dict["instances"].gt_classes],
            )
        else:
            vis = vis.overlay_instances(
                boxes=None,
                masks=None,
                labels=None,
            )
        vis_gt = vis.get_image()

        if "instances_phrase" in dataset_dict:
            vis = Visualizer(img, metadata=metadata)
            vis = vis.overlay_instances(
                boxes=dataset_dict["instances_phrase"].gt_boxes,
                masks=dataset_dict["instances_phrase"].gt_masks
                if dataset_dict["instances_phrase"].has("gt_masks")
                else None,
                labels=dataset_dict["instances_phrase"].phrases,
            )
            vis_phrase = vis.get_image()
            vis_gt = np.concatenate((vis_gt, vis_phrase), axis=1)

        if "captions" in dataset_dict:
            vis = Visualizer(img, metadata=metadata)
            vis = vis.overlay_instances(
                boxes=Boxes(
                    np.array(
                        [
                            [
                                0 + i * 20,
                                0 + i * 20,
                                image_shape[1] - 1 - i * 20,
                                image_shape[0] - 1 - i * 20,
                            ]
                            for i in range(len(dataset_dict["captions"]))
                        ]
                    )
                ),
                masks=None,
                labels=dataset_dict["captions"],
            )
            vis_cap = vis.get_image()
            vis_gt = np.concatenate((vis_gt, vis_cap), axis=1)

        if "sem_seg" in dataset_dict:
            vis = Visualizer(img, metadata=metadata)
            vis = vis.draw_sem_seg(dataset_dict["sem_seg"], area_threshold=0, alpha=0.5)
            vis_sem_gt = vis.get_image()
            vis_gt = np.concatenate((vis_gt, vis_sem_gt), axis=1)

        concat = np.concatenate((vis_gt, img), axis=1)

        image_name = os.path.basename(dataset_dict["file_name"]).split(".")[0]

        save_path = os.path.join(
            self.output_dir,
            prefix
            + str(self.iter)
            + "_"
            + image_name
            + "_g"
            + str(comm.get_rank())
            + suffix
            + ".png",
        )
        concat = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, concat)

        return

        import pickle

        save_path = os.path.join(
            self.output_dir,
            prefix
            + str(self.iter)
            + "_"
            + str(dataset_dict["image_id"])
            + "_g"
            + str(comm.get_rank())
            + suffix
            + ".pkl",
        )
        with open(save_path, "wb") as save_file:
            pickle.dump(dataset_dict, save_file)
