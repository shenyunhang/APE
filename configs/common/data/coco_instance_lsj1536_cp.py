import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import DatasetMapper, build_detection_test_loader, get_detection_dataset_dicts
from detectron2.evaluation import COCOEvaluator
from omegaconf import OmegaConf
from ape.data import (
    DatasetMapper_copypaste,
    build_detection_train_loader_copypaste,
    get_detection_dataset_dicts_copypaste,
)

image_size = 1536

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader_copypaste)(
    dataset=L(get_detection_dataset_dicts_copypaste)(names=["coco_2017_train"], copypastes=[True]),
    dataset_bg=L(get_detection_dataset_dicts)(names=["coco_2017_train"]),
    mapper=L(DatasetMapper_copypaste)(
        is_train=True,
        augmentations=[
            L(T.RandomFlip)(horizontal=True),  # flip first
            L(T.ResizeScale)(
                min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
            ),
            L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
        ],
        augmentations_d2=[],
        augmentations_aa=[],
        augmentations_lsj=[],
        augmentations_type=[],
        image_format="RGB",
        use_instance_mask=True,
        recompute_boxes=True,
        output_dir=None,
        vis_period=12800,
        dataset_names=["coco_2017_train"],
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    max_dets_per_image=100,
)
