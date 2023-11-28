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

image_size = 1536

odinw_dataset_metas = [
    "odinw_AerialMaritimeDrone_large_train",
    "odinw_Aquarium_Aquarium_Combined.v2-raw-1024.coco_train",
    "odinw_CottontailRabbits_train",
    "odinw_EgoHands_generic_train",
    "odinw_NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco_train",
    "odinw_Packages_Raw_train",
    "odinw_PascalVOC_train",
    "odinw_pistols_export_train",
    "odinw_pothole_train",
    "odinw_Raccoon_Raccoon.v2-raw.coco_train",
    "odinw_ShellfishOpenImages_raw_train",
    "odinw_thermalDogsAndPeople_train",
    "odinw_VehiclesOpenImages_416x416_train",
]

dataloader.train = [
    L(build_detection_train_loader_multi_dataset)(
        dataset=L(get_detection_dataset_dicts_multi_dataset)(
            names=(dataset_name,),
            filter_emptys=[True],
            dataloader_id=dataloader_id,
            reduce_memory=True,
            reduce_memory_size=1e6,
        ),
        mapper=L(DatasetMapper_detr_panoptic)(
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
        total_batch_size=16,
        total_batch_size_list=[16],
        aspect_ratio_grouping=True,
        num_workers=16,
        num_datasets=1,
    )
    for dataloader_id, dataset_name in enumerate(odinw_dataset_metas)
]

odinw_test_dataset_names = [
    "odinw_AerialMaritimeDrone_large_test",
    "odinw_Aquarium_Aquarium_Combined.v2-raw-1024.coco_test",
    "odinw_CottontailRabbits_test",
    "odinw_EgoHands_generic_test",
    "odinw_NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco_test",
    "odinw_Packages_Raw_test",
    "odinw_PascalVOC_val",
    "odinw_pistols_export_test",
    "odinw_pothole_test",
    "odinw_Raccoon_Raccoon.v2-raw.coco_test",
    "odinw_ShellfishOpenImages_raw_test",
    "odinw_thermalDogsAndPeople_test",
    "odinw_VehiclesOpenImages_416x416_test",
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
    for name in odinw_test_dataset_names
]

dataloader.name_prompt_fusion_text = [
    True,
    True,
    False,
    False,
    True,
    True,
    False,
    False,
    True,
    True,
    False,
    True,
    False,
]

dataloader.select_box_nums_for_evaluation_list = [
    300,
    300,
    300,
    300,
    300,
    300,
    300,
    300,
    300,
    300,
    300,
    300,
    300,
]

dataloader.evaluators = [
    L(COCOEvaluator)(
        dataset_name=name,
        tasks=("bbox",),
    )
    for name in odinw_test_dataset_names
]
