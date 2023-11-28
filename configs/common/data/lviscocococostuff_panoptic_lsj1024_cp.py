import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation import LVISEvaluator, SemSegEvaluator
from omegaconf import OmegaConf
from ape.data import (
    DatasetMapper_detr_panoptic_copypaste,
    build_detection_train_loader_copypaste,
    get_detection_dataset_dicts_copypaste,
)

dataloader = OmegaConf.create()

image_size = 1024

dataloader.train = L(build_detection_train_loader_copypaste)(
    dataset=L(get_detection_dataset_dicts_copypaste)(
        names="lvis_v1_train+coco_panoptic_separated", copypastes=[True]
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
        dataset_names=["lvis_v1_train+coco_panoptic_separated"],
    ),
    sampler=L(RepeatFactorTrainingSampler)(
        repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
            dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001
        )
    ),
    sampler_bg=L(RepeatFactorTrainingSampler)(
        repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
            dataset_dicts="${dataloader.train.dataset_bg}", repeat_thresh=0.001
        ),
    ),
    total_batch_size=16,
    aspect_ratio_grouping=True,
    num_workers=16,
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
