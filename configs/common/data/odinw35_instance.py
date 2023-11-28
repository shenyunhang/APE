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

odinw_dataset_metas = [
    "odinw_AerialMaritimeDrone_large_train",
    "odinw_AerialMaritimeDrone_tiled_train",
    "odinw_AmericanSignLanguageLetters_American_Sign_Language_Letters.v1-v1.coco_train",
    "odinw_Aquarium_Aquarium_Combined.v2-raw-1024.coco_train",
    "odinw_BCCD_BCCD.v3-raw.coco_train",
    "odinw_boggleBoards_416x416AutoOrient_export_train",
    "odinw_brackishUnderwater_960x540_train",
    "odinw_ChessPieces_Chess_Pieces.v23-raw.coco_train",
    "odinw_CottontailRabbits_train",
    "odinw_dice_mediumColor_export_train",
    "odinw_DroneControl_Drone_Control.v3-raw.coco_train",
    "odinw_EgoHands_generic_train",
    "odinw_EgoHands_specific_train",
    "odinw_HardHatWorkers_raw_train",
    "odinw_MaskWearing_raw_train",
    "odinw_MountainDewCommercial_train",
    "odinw_NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco_train",
    "odinw_openPoetryVision_512x512_train",
    "odinw_OxfordPets_by-breed_train",
    "odinw_OxfordPets_by-species_train",
    "odinw_Packages_Raw_train",
    "odinw_PascalVOC_train",
    "odinw_pistols_export_train",
    "odinw_PKLot_640_train",
    "odinw_plantdoc_416x416_train",
    "odinw_pothole_train",
    "odinw_Raccoon_Raccoon.v2-raw.coco_train",
    "odinw_selfdrivingCar_fixedLarge_export_train",
    "odinw_ShellfishOpenImages_raw_train",
    "odinw_ThermalCheetah_train",
    "odinw_thermalDogsAndPeople_train",
    "odinw_UnoCards_raw_train",
    "odinw_VehiclesOpenImages_416x416_train",
    "odinw_websiteScreenshots_train",
    "odinw_WildfireSmoke_train",
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
    "odinw_AerialMaritimeDrone_tiled_test",
    "odinw_AmericanSignLanguageLetters_American_Sign_Language_Letters.v1-v1.coco_test",
    "odinw_Aquarium_Aquarium_Combined.v2-raw-1024.coco_test",
    "odinw_BCCD_BCCD.v3-raw.coco_test",
    "odinw_boggleBoards_416x416AutoOrient_export_test",
    "odinw_brackishUnderwater_960x540_test",
    "odinw_ChessPieces_Chess_Pieces.v23-raw.coco_test",
    "odinw_CottontailRabbits_test",
    "odinw_dice_mediumColor_export_test",
    "odinw_DroneControl_Drone_Control.v3-raw.coco_test",
    "odinw_EgoHands_generic_test",
    "odinw_EgoHands_specific_test",
    "odinw_HardHatWorkers_raw_test",
    "odinw_MaskWearing_raw_test",
    "odinw_MountainDewCommercial_test",
    "odinw_NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco_test",
    "odinw_openPoetryVision_512x512_test",
    "odinw_OxfordPets_by-breed_test",
    "odinw_OxfordPets_by-species_test",
    "odinw_Packages_Raw_test",
    "odinw_PascalVOC_val",
    "odinw_pistols_export_test",
    "odinw_PKLot_640_test",
    "odinw_plantdoc_416x416_test",
    "odinw_pothole_test",
    "odinw_Raccoon_Raccoon.v2-raw.coco_test",
    "odinw_selfdrivingCar_fixedLarge_export_test",
    "odinw_ShellfishOpenImages_raw_test",
    "odinw_ThermalCheetah_test",
    "odinw_thermalDogsAndPeople_test",
    "odinw_UnoCards_raw_test",
    "odinw_VehiclesOpenImages_416x416_test",
    "odinw_websiteScreenshots_test",
    "odinw_WildfireSmoke_test",
]

dataloader.tests = [
    L(build_detection_test_loader)(
        dataset=L(get_detection_dataset_dicts)(names=name, filter_empty=False),
        mapper=L(DatasetMapper)(
            is_train=False,
            augmentations=[
                L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
            ],
            image_format="RGB",
        ),
        num_workers=4,
    )
    for name in odinw_test_dataset_names
]

dataloader.name_prompt_fusion_text = [
    True,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    False,
    True,
    False,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    False,
    False,
    True,
    True,
    True,
    True,
    False,
    False,
    True,
    True,
    False,
    True,
    True,
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
    1,
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
