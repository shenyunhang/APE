import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import DatasetMapper, build_detection_test_loader, get_detection_dataset_dicts
from detectron2.evaluation import COCOEvaluator
from omegaconf import OmegaConf

dataloader = OmegaConf.create()

image_size = 1024

odinw_test_dataset_names = [
    "odinw_PascalVOC_val",
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
    False,
]

dataloader.select_box_nums_for_evaluation_list = [
    300,
]

dataloader.evaluators = [
    L(COCOEvaluator)(
        dataset_name=name,
        tasks=("bbox",),
    )
    for name in odinw_test_dataset_names
]
