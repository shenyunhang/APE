import random

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation import COCOEvaluator, LVISEvaluator, SemSegEvaluator
from omegaconf import OmegaConf
from ape.data import (
    DatasetMapper_detr_panoptic_copypaste,
    build_detection_train_loader_multi_dataset_copypaste,
    get_detection_dataset_dicts_multi_dataset_copypaste,
)
from ape.data.samplers import MultiDatasetTrainingSampler
from ape.evaluation import RefCOCOEvaluator
from ape.evaluation.oideval import OIDEvaluator

dataloader = OmegaConf.create()

image_size = 1024

dataloader.train = L(build_detection_train_loader_multi_dataset_copypaste)(
    dataset=L(get_detection_dataset_dicts_multi_dataset_copypaste)(
        names=(
            "lvis_v1_train+coco_panoptic_separated",
            "objects365_train_fixname",
            "openimages_v6_train_bbox_nogroup",
            "visualgenome_77962_box_and_region",
            "sa1b_6m",
            "refcoco-mixed_group-by-image",
            "gqa_region_train",
            "phrasecut_train",
            "flickr30k_separateGT_train",
        ),
        filter_emptys=[True, True, True, True, False, True, True, True, True],
        copypastes=[True, False, False, False, False, False, False, False, False],
        reduce_memory=True,
        reduce_memory_size=1e6,
    ),
    dataset_bg=L(get_detection_dataset_dicts)(names=["lvis_v1_train+coco_panoptic_separated"]),
    mapper=L(DatasetMapper_detr_panoptic_copypaste)(
        is_train=True,
        augmentations=[
            L(T.RandomFlip)(horizontal=True),  # flip first
            L(T.ResizeScale)(
                min_scale=0.1, max_scale=1.0, target_height=image_size, target_width=image_size
            ),
            L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
        ],
        augmentations_with_crop=[
            L(T.RandomFlip)(horizontal=True),  # flip first
            L(T.ResizeScale)(
                min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
            ),
            L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
        ],
        image_format="RGB",
        use_instance_mask=True,
        recompute_boxes=True,
        instance_mask_format="bitmask",
        ignore_label=MetadataCatalog.get("coco_2017_train_panoptic_stuffonly").ignore_label,
        stuff_classes_offset=1203,
        stuff_classes_decomposition=True,
        output_dir=None,
        vis_period=12800,
        dataset_names="${..dataset.names}",
        max_num_phrase=128,
        nms_thresh_phrase=0.6,
    ),
    sampler=lambda dataset_dicts: MultiDatasetTrainingSampler(
        repeat_factors=MultiDatasetTrainingSampler.get_repeat_factors(
            dataset_dicts=dataset_dicts,
            num_datasets=9,
            dataset_ratio=[1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1],
            use_rfs=[True, True, True, False, False, False, False, False, False],
            use_cas=[False, False, False, False, False, False, False, False, False],
            repeat_thresh=0.001,
            cas_lambda=1.0,
        ),
        seed=random.randint(0, 2**31),
    ),
    sampler_bg=lambda dataset_dicts: RepeatFactorTrainingSampler(
        repeat_factors=RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts=dataset_dicts, repeat_thresh=0.001
        ),
        seed=random.randint(0, 2**31),
    ),
    total_batch_size=16,
    total_batch_size_list=[16, 16, 16, 16, 16, 16, 16, 16, 16],
    aspect_ratio_grouping=True,
    num_workers=2,
    num_datasets=9,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="lvis_v1_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(LVISEvaluator)(
    dataset_name="${..test.dataset.names}",
    max_dets_per_image=300,
)

dataloader.tests = [
    L(build_detection_test_loader)(
        dataset=L(get_detection_dataset_dicts)(
            names="coco_2017_val_panoptic_stuffonly", filter_empty=False
        ),
        mapper=L(DatasetMapper)(
            is_train=False,
            augmentations=[
                L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
            ],
            image_format="${....train.mapper.image_format}",
        ),
        num_workers=4,
    ),
]

dataloader.evaluators = [
    L(SemSegEvaluator)(
        dataset_name="coco_2017_val_panoptic_stuffonly",
    ),
]

dataloader.tests += [
    L(build_detection_test_loader)(
        dataset=L(get_detection_dataset_dicts)(names="lvis_v1_minival", filter_empty=False),
        mapper=L(DatasetMapper)(
            is_train=False,
            augmentations=[
                L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
            ],
            image_format="${....train.mapper.image_format}",
        ),
        num_workers=4,
    )
]

dataloader.evaluators += [
    L(LVISEvaluator)(
        dataset_name="lvis_v1_minival",
        max_dets_per_image=300,
    )
]

dataloader.tests += [
    L(build_detection_test_loader)(
        dataset=L(get_detection_dataset_dicts)(
            names="objects365_minival_fixname", filter_empty=False
        ),
        mapper=L(DatasetMapper)(
            is_train=False,
            augmentations=[
                L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
            ],
            image_format="${....train.mapper.image_format}",
        ),
        num_workers=4,
    ),
]

dataloader.evaluators += [
    L(COCOEvaluator)(
        dataset_name="objects365_minival_fixname",
        tasks=("bbox",),
    ),
]

dataloader.tests += [
    L(build_detection_test_loader)(
        dataset=L(get_detection_dataset_dicts)(names="objects365_val_fixname", filter_empty=False),
        mapper=L(DatasetMapper)(
            is_train=False,
            augmentations=[
                L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
            ],
            image_format="${....train.mapper.image_format}",
        ),
        num_workers=4,
    ),
]

dataloader.evaluators += [
    L(COCOEvaluator)(
        dataset_name="objects365_val_fixname",
        tasks=("bbox",),
    ),
]

dataloader.tests += [
    L(build_detection_test_loader)(
        dataset=L(get_detection_dataset_dicts)(names="openimages_v6_val_bbox", filter_empty=False),
        mapper=L(DatasetMapper)(
            is_train=False,
            augmentations=[
                L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
            ],
            image_format="${....train.mapper.image_format}",
        ),
        num_workers=4,
    ),
]

dataloader.evaluators += [
    L(OIDEvaluator)(
        dataset_name="openimages_v6_val_bbox",
    ),
]



refcoco_test_dataset_names = [
    "refcoco-unc-val",
    "refcoco-unc-testA",
    "refcoco-unc-testB",
    "refcocoplus-unc-val",
    "refcocoplus-unc-testA",
    "refcocoplus-unc-testB",
    "refcocog-google-val",
    "refcocog-umd-val",
    "refcocog-umd-test",
]
dataloader.tests += [
    L(build_detection_test_loader)(
        dataset=L(get_detection_dataset_dicts)(names=name, filter_empty=False),
        mapper=L(DatasetMapper)(
            is_train=False,
            augmentations=[
                L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
            ],
            image_format="${....train.mapper.image_format}",
        ),
        num_workers=4,
    )
    for name in refcoco_test_dataset_names
]

dataloader.evaluators += [
    L(RefCOCOEvaluator)(
        dataset_name=name,
    )
    for name in refcoco_test_dataset_names
]
