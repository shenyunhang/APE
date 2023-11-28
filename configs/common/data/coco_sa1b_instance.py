import random

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator, COCOPanopticEvaluator, SemSegEvaluator
from omegaconf import OmegaConf
from ape.data import (
    DatasetMapper_detr_panoptic,
    build_detection_train_loader_multi_dataset,
    get_detection_dataset_dicts_multi_dataset,
)
from ape.data.samplers import MultiDatasetTrainingSampler

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader_multi_dataset)(
    dataset=L(get_detection_dataset_dicts_multi_dataset)(
        names=(
            "coco_2017_train",
            "sa1b_4m",
        ),
        filter_emptys=[True, False],
    ),
    mapper=L(DatasetMapper_detr_panoptic)(
        is_train=True,
        augmentations=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentations_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        image_format="RGB",
        use_instance_mask=True,
        recompute_boxes=True,
        instance_mask_format="bitmask",
        ignore_label=MetadataCatalog.get("coco_2017_train_panoptic_stuffonly").ignore_label,
        stuff_classes_offset=80,
        dataset_names=["coco_2017_train", "sa1b"],
    ),
    sampler=lambda dataset_dicts: MultiDatasetTrainingSampler(
        repeat_factors=MultiDatasetTrainingSampler.get_repeat_factors(
            dataset_dicts=dataset_dicts,
            num_datasets=2,
            dataset_ratio=[1, 1],
            use_rfs=[False, False],
            use_cas=[False, False],
            repeat_thresh=0.001,
            cas_lambda=1.0,
        ),
        seed=random.randint(0, 2**31),
    ),
    total_batch_size=16,
    total_batch_size_list=[16, 16],
    num_workers=4,
    num_datasets=2,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
