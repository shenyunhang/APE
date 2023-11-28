import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from omegaconf import OmegaConf
from ape.data import (
    DatasetMapper_detr_instance,
    build_detection_train_loader_multi_dataset,
    get_detection_dataset_dicts_multi_dataset,
)
from ape.evaluation import RefCOCOEvaluator

image_size = 1024

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader_multi_dataset)(
    dataset=L(get_detection_dataset_dicts_multi_dataset)(
        names=("refcoco-mixed_group-by-image",), filter_emptys=[True]
    ),
    mapper=L(DatasetMapper_detr_instance)(
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
        dataset_names=("refcoco-mixed_group-by-image",),
        max_num_phrase=128,
        nms_thresh_phrase=0.6,
    ),
    total_batch_size=16,
    total_batch_size_list=[16],
    num_workers=4,
    num_datasets=1,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="refcoco-unc-val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(RefCOCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)

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
dataloader.tests = [
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
    for name in refcoco_test_dataset_names[1:]
]

dataloader.evaluators = [
    L(RefCOCOEvaluator)(
        dataset_name=name,
    )
    for name in refcoco_test_dataset_names[1:]
]
