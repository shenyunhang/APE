import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from omegaconf import OmegaConf
from ape.data import (
    DatasetMapper_detr_panoptic,
    DatasetMapper_detr_panoptic_copypaste,
    build_detection_train_loader_multi_dataset,
    build_detection_train_loader_multi_dataset_copypaste,
    get_detection_dataset_dicts_multi_dataset,
    get_detection_dataset_dicts_multi_dataset_copypaste,
)

dataloader = OmegaConf.create()

image_size = 1024

seginw_dataset_metas = [
    "seginw_Elephants_train",
    "seginw_Hand-Metal_train",
    "seginw_Watermelon_train",
    "seginw_House-Parts_train",
    "seginw_HouseHold-Items_train",
    "seginw_Strawberry_train",
    "seginw_Fruits_train",
    "seginw_Nutterfly-Squireel_train",
    "seginw_Hand_train",
    "seginw_Garbage_train",
    "seginw_Chicken_train",
    "seginw_Rail_train",
    "seginw_Airplane-Parts_train",
    "seginw_Brain-Tumor_train",
    "seginw_Poles_train",
    "seginw_Electric-Shaver_train",
    "seginw_Bottles_train",
    "seginw_Toolkits_train",
    "seginw_Trash_train",
    "seginw_Salmon-Fillet_train",
    "seginw_Puppies_train",
    "seginw_Tablets_train",
    "seginw_Phones_train",
    "seginw_Cows_train",
    "seginw_Ginger-Garlic_train",
]

dataloader.train = [
    L(build_detection_train_loader_multi_dataset_copypaste)(
        dataset=L(get_detection_dataset_dicts_multi_dataset_copypaste)(
            names=(dataset_name,),
            filter_emptys=[True],
            copypastes=[True],
            dataloader_id=dataloader_id,
            reduce_memory=True,
            reduce_memory_size=1e6,
        ),
        dataset_bg=L(get_detection_dataset_dicts)(
            names=(dataset_name,),
            filter_empty=True,
        ),
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
            ignore_label=MetadataCatalog.get(dataset_name).get("ignore_label", None),
            stuff_classes_offset=len(MetadataCatalog.get(dataset_name).get("thing_classes", [])),
            stuff_classes_decomposition=True,
            output_dir=None,
            vis_period=12800,
            dataset_names=(dataset_name,),
            max_num_phrase=128,
            nms_thresh_phrase=0.6,
        ),
        sampler=None,
        sampler_bg=None,
        total_batch_size=16,
        total_batch_size_list=[16],
        aspect_ratio_grouping=True,
        num_workers=16,
        num_datasets=1,
    )
    for dataloader_id, dataset_name in enumerate(seginw_dataset_metas)
]

seginw_test_dataset_names = [
    "seginw_Elephants_val",
    "seginw_Hand-Metal_val",
    "seginw_Watermelon_val",
    "seginw_House-Parts_val",
    "seginw_HouseHold-Items_val",
    "seginw_Strawberry_val",
    "seginw_Fruits_val",
    "seginw_Nutterfly-Squireel_val",
    "seginw_Hand_val",
    "seginw_Garbage_val",
    "seginw_Chicken_val",
    "seginw_Rail_val",
    "seginw_Airplane-Parts_val",
    "seginw_Brain-Tumor_val",
    "seginw_Poles_val",
    "seginw_Electric-Shaver_val",
    "seginw_Bottles_val",
    "seginw_Toolkits_val",
    "seginw_Trash_val",
    "seginw_Salmon-Fillet_val",
    "seginw_Puppies_val",
    "seginw_Tablets_val",
    "seginw_Phones_val",
    "seginw_Cows_val",
    "seginw_Ginger-Garlic_val",
]

dataloader.tests = [
    L(build_detection_test_loader)(
        dataset=L(get_detection_dataset_dicts)(names=name, filter_empty=False),
        mapper=L(DatasetMapper)(
            is_train=False,
            augmentations=[
                L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
            ],
            image_format="RGB",
        ),
        num_workers=4,
    )
    for name in seginw_test_dataset_names
]

dataloader.evaluators = [
    L(COCOEvaluator)(
        dataset_name=name,
    )
    for name in seginw_test_dataset_names
]
