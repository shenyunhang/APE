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
    DatasetMapper_detr_instance_exp,
    build_detection_train_loader_multi_dataset,
    get_detection_dataset_dicts_multi_dataset,
)
from ape.evaluation import RefCOCOEvaluator

dataloader = OmegaConf.create()

image_size = 1024

dataloader.train = L(build_detection_train_loader_multi_dataset)(
    dataset=L(get_detection_dataset_dicts_multi_dataset)(
        names=("phrasecut_train",), filter_emptys=[True]
    ),
    mapper=L(DatasetMapper_detr_instance_exp)(
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
        dataset_names=("phrasecut_train",),
        max_num_phrase=256,
        nms_thresh_phrase=0.6,
    ),
    total_batch_size=16,
    total_batch_size_list=[16],
    num_workers=4,
    num_datasets=1,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="phrasecut_val", filter_empty=False),
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
