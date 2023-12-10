import gc
import multiprocessing as mp
import os
import shutil
import sys
import time
from os import path

import cv2
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

import ape
import detectron2.data.transforms as T
import gradio as gr
from ape.model_zoo import get_config_file
from demo_lazy import get_parser, setup_cfg
from detectron2.config import CfgNode
from detectron2.data.detection_utils import read_image
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.utils.logger import setup_logger
from predictor_lazy import VisualizationDemo

this_dir = path.dirname(path.abspath(__file__))

# os.system("git clone https://github.com/shenyunhang/APE.git")
# os.system("python3.10 -m pip install -e APE/")

example_list = [
    [
        this_dir + "/examples/Totoro01.png",
        # "Sky, Water, Tree, The biggest Chinchilla, The older girl wearing skirt on branch, Grass",
        "Girl with hat",
        # 0.05,
        0.25,
        ["object detection", "instance segmentation"],
    ],
    [
        this_dir + "/examples/Totoro01.png",
        "Sky, Water, Tree, Chinchilla, Grass, Girl",
        0.15,
        ["semantic segmentation"],
    ],
    [
        this_dir + "/examples/199_3946193540.jpg",
        "chess piece of horse head",
        0.30,
        ["object detection", "instance segmentation"],
    ],
    [
        this_dir + "/examples/TheGreatWall.jpg",
        "The Great Wall",
        0.1,
        ["semantic segmentation"],
    ],
    [
        this_dir + "/examples/Pisa.jpg",
        "Pisa",
        0.01,
        ["object detection", "instance segmentation"],
    ],
    [
        this_dir + "/examples/SolvayConference1927.jpg",
        # "Albert Einstein, Madame Curie",
        "Madame Curie",
        # 0.01,
        0.03,
        ["object detection", "instance segmentation"],
    ],
    [
        this_dir + "/examples/Transformers.webp",
        "Optimus Prime",
        0.11,
        ["object detection", "instance segmentation"],
    ],
    [
        this_dir + "/examples/Terminator3.jpg",
        "Humanoid Robot",
        0.10,
        ["object detection", "instance segmentation"],
    ],
    [
        this_dir + "/examples/MatrixRevolutionForZion.jpg",
        """machine killer with gun in fighting,
donut with colored granules on the surface,
railings being crossed by horses, 
a horse running or jumping,
equestrian rider's helmet,
outdoor dog led by rope, 
a dog being touched, 
clothed dog, 
basketball in hand, 
a basketball player with both feet off the ground, 
player with basketball in the hand, 
spoon on the plate, 
coffee cup with coffee, 
the nearest dessert to the coffee cup, 
the bartender who is mixing wine, 
a bartender in a suit, 
wine glass with wine, 
a person in aprons, 
pot with food, 
a knife being used to cut vegetables, 
striped sofa in the room, 
a sofa with pillows on it in the room, 
lights on in the room, 
an indoor lying pet, 
a cat on the sofa, 
one pet looking directly at the camera indoors, 
a bed with patterns in the room, 
the lamp on the table beside the bed, 
pillow placed at the head of the bed, 
a blackboard full of words in the classroom, 
child sitting at desks in the classroom, 
a person standing in front of bookshelves in the library, 
the table someone is using in the library, 
a person who touches books in the library, 
a person standing in front of the cake counter, 
a square plate full of cakes, 
a cake decorated with cream, 
hot dog with vegetables, 
hot dog with sauce on the surface, 
red sausage, 
flowerpot with flowers potted inside, 
monochrome flowerpot, 
a flowerpot filled with black soil, 
apple growing on trees, 
red complete apple, 
apple with a stalk, 
a woman brushing her teeth, 
toothbrush held by someone, 
toilet brush with colored bristles, 
a customer whose hair is being cut by barber, 
a barber at work, 
cloth covering the barber, 
shopping cart pushed by people in the supermarket, 
shopping cart with people in the supermarket, 
shopping cart full of goods, 
a child wearing a mask, 
refrigerator with fruit, 
a drink bottle in the refrigerator, 
refrigerator with more than two doors, 
a watch placed on a table or cloth, 
a watch with three or more watch hands can be seen, 
a watch with one or more small dials, 
clothes hanger, 
a piece of clothing hanging on the hanger, 
a piece of clothing worn on plastic models, 
leather bag with glossy surface, 
backpack, 
open package, 
a fish held by people, 
a person who is fishing with a fishing rod, 
a fisherman standing on the shore with his body soaked in water, camera hold on someone's shoulder,
a person being interviewed, 
a person with microphone hold in hand,
        """,
        0.20,
        ["object detection", "instance segmentation"],
    ],
    [
        this_dir + "/examples/094_56726435.jpg",
        # "donut with colored granules on the surface",
        """donut with colored granules on the surface,
railings being crossed by horses, 
a horse running or jumping,
equestrian rider's helmet,
outdoor dog led by rope, 
a dog being touched, 
clothed dog, 
basketball in hand, 
a basketball player with both feet off the ground, 
player with basketball in the hand, 
spoon on the plate, 
coffee cup with coffee, 
the nearest dessert to the coffee cup, 
the bartender who is mixing wine, 
a bartender in a suit, 
wine glass with wine, 
a person in aprons, 
pot with food, 
a knife being used to cut vegetables, 
striped sofa in the room, 
a sofa with pillows on it in the room, 
lights on in the room, 
an indoor lying pet, 
a cat on the sofa, 
one pet looking directly at the camera indoors, 
a bed with patterns in the room, 
the lamp on the table beside the bed, 
pillow placed at the head of the bed, 
a blackboard full of words in the classroom, 
a blackboard or whiteboard with something pasted, 
child sitting at desks in the classroom, 
a person standing in front of bookshelves in the library, 
the table someone is using in the library, 
a person who touches books in the library, 
a person standing in front of the cake counter, 
a square plate full of cakes, 
a cake decorated with cream, 
hot dog with vegetables, 
hot dog with sauce on the surface, 
red sausage, 
flowerpot with flowers potted inside, 
monochrome flowerpot, 
a flowerpot filled with black soil, 
apple growing on trees, 
red complete apple, 
apple with a stalk, 
a woman brushing her teeth, 
toothbrush held by someone, 
toilet brush with colored bristles, 
a customer whose hair is being cut by barber, 
a barber at work, 
cloth covering the barber, 
a plastic toy, 
a plush toy, 
a humanoid toy, 
shopping cart pushed by people in the supermarket, 
shopping cart with people in the supermarket, 
shopping cart full of goods, 
a child wearing a mask, 
a mask on face with half a face exposed, 
a mask on face with only eyes exposed, 
refrigerator with fruit, 
a drink bottle in the refrigerator, 
refrigerator with more than two doors, 
a watch placed on a table or cloth, 
a watch with three or more watch hands can be seen, 
a watch with one or more small dials, 
clothes hanger, 
a piece of clothing hanging on the hanger, 
a piece of clothing worn on plastic models, 
leather bag with glossy surface, 
backpack, 
open package, 
a fish held by people, 
a person who is fishing with a fishing rod, 
a fisherman standing on the shore with his body soaked in water, camera hold on someone's shoulder,
a person being interviewed, 
a person with microphone hold in hand,
        """,
        0.50,
        ["object detection", "instance segmentation"],
    ],
    [
        this_dir + "/examples/013_438973263.jpg",
        # "a male lion with a mane",
        """a male lion with a mane,
railings being crossed by horses, 
a horse running or jumping,
equestrian rider's helmet,
outdoor dog led by rope, 
a dog being touched, 
clothed dog, 
basketball in hand, 
a basketball player with both feet off the ground, 
player with basketball in the hand, 
spoon on the plate, 
coffee cup with coffee, 
the nearest dessert to the coffee cup, 
the bartender who is mixing wine, 
a bartender in a suit, 
wine glass with wine, 
a person in aprons, 
pot with food, 
a knife being used to cut vegetables, 
striped sofa in the room, 
a sofa with pillows on it in the room, 
lights on in the room, 
an indoor lying pet, 
a cat on the sofa, 
one pet looking directly at the camera indoors, 
a bed with patterns in the room, 
the lamp on the table beside the bed, 
pillow placed at the head of the bed, 
a blackboard full of words in the classroom, 
a blackboard or whiteboard with something pasted, 
child sitting at desks in the classroom, 
a person standing in front of bookshelves in the library, 
the table someone is using in the library, 
a person who touches books in the library, 
a person standing in front of the cake counter, 
a square plate full of cakes, 
a cake decorated with cream, 
hot dog with vegetables, 
hot dog with sauce on the surface, 
red sausage, 
flowerpot with flowers potted inside, 
monochrome flowerpot, 
a flowerpot filled with black soil, 
apple growing on trees, 
red complete apple, 
apple with a stalk, 
a woman brushing her teeth, 
toothbrush held by someone, 
toilet brush with colored bristles, 
a customer whose hair is being cut by barber, 
a barber at work, 
cloth covering the barber, 
a plastic toy, 
a plush toy, 
a humanoid toy, 
shopping cart pushed by people in the supermarket, 
shopping cart with people in the supermarket, 
shopping cart full of goods, 
a child wearing a mask, 
a mask on face with half a face exposed, 
a mask on face with only eyes exposed, 
refrigerator with fruit, 
a drink bottle in the refrigerator, 
refrigerator with more than two doors, 
a watch placed on a table or cloth, 
a watch with three or more watch hands can be seen, 
a watch with one or more small dials, 
clothes hanger, 
a piece of clothing hanging on the hanger, 
a piece of clothing worn on plastic models, 
leather bag with glossy surface, 
backpack, 
open package, 
a fish held by people, 
a person who is fishing with a fishing rod, 
a fisherman standing on the shore with his body soaked in water, camera hold on someone's shoulder,
a person being interviewed, 
a person with microphone hold in hand,
        """,
        # 0.25,
        0.50,
        ["object detection", "instance segmentation"],
    ],
]

ckpt_repo_id = "shenyunhang/APE"


def setup_model(name):
    gc.collect()
    torch.cuda.empty_cache()

    if save_memory:
        pass
    else:
        return

    for key, demo in all_demo.items():
        if key == name:
            demo.predictor.model.to(running_device)
        else:
            demo.predictor.model.to("cpu")

    gc.collect()
    torch.cuda.empty_cache()


def run_on_image_A(input_image_path, input_text, score_threshold, output_type):
    logger.info("run_on_image")

    setup_model("APE_A")
    demo = all_demo["APE_A"]
    cfg = all_cfg["APE_A"]
    demo.predictor.model.model_vision.test_score_thresh = score_threshold

    return run_on_image(
        input_image_path,
        input_text,
        output_type,
        demo,
        cfg,
    )


def run_on_image_C(input_image_path, input_text, score_threshold, output_type):
    logger.info("run_on_image_C")

    setup_model("APE_C")
    demo = all_demo["APE_C"]
    cfg = all_cfg["APE_C"]
    demo.predictor.model.model_vision.test_score_thresh = score_threshold

    return run_on_image(
        input_image_path,
        input_text,
        output_type,
        demo,
        cfg,
    )


def run_on_image_D(input_image_path, input_text, score_threshold, output_type):
    logger.info("run_on_image_D")

    setup_model("APE_D")
    demo = all_demo["APE_D"]
    cfg = all_cfg["APE_D"]
    demo.predictor.model.model_vision.test_score_thresh = score_threshold

    return run_on_image(
        input_image_path,
        input_text,
        output_type,
        demo,
        cfg,
    )


def run_on_image_comparison(input_image_path, input_text, score_threshold, output_type):
    logger.info("run_on_image_comparison")

    r = []
    for key in all_demo.keys():
        logger.info("run_on_image_comparison {}".format(key))
        setup_model(key)
        demo = all_demo[key]
        cfg = all_cfg[key]
        demo.predictor.model.model_vision.test_score_thresh = score_threshold

        img, _ = run_on_image(
            input_image_path,
            input_text,
            output_type,
            demo,
            cfg,
        )
        r.append(img)

    return r


def run_on_image(
    input_image_path,
    input_text,
    output_type,
    demo,
    cfg,
):
    with_box = False
    with_mask = False
    with_sseg = False
    if "object detection" in output_type:
        with_box = True
    if "instance segmentation" in output_type:
        with_mask = True
    if "semantic segmentation" in output_type:
        with_sseg = True

    if isinstance(input_image_path, dict):
        input_mask_path = input_image_path["mask"]
        input_image_path = input_image_path["image"]
        print("input_image_path", input_image_path)
        print("input_mask_path", input_mask_path)
    else:
        input_mask_path = None

    print("input_text", input_text)

    if isinstance(cfg, CfgNode):
        input_format = cfg.INPUT.FORMAT
    else:
        if "model_vision" in cfg.model:
            input_format = cfg.model.model_vision.input_format
        else:
            input_format = cfg.model.input_format

    input_image = read_image(input_image_path, format="BGR")
    # img = cv2.imread(input_image_path)
    # cv2.imwrite("tmp.jpg", img)
    # # input_image = read_image("tmp.jpg", format=input_format)
    # input_image = read_image("tmp.jpg", format="BGR")

    if input_mask_path is not None:
        input_mask = read_image(input_mask_path, "L").squeeze(2)
        print("input_mask", input_mask)
        print("input_mask", input_mask.shape)
    else:
        input_mask = None

    if not with_box and not with_mask and not with_sseg:
        return input_image[:, :, ::-1]

    if input_image.shape[0] > 1024 or input_image.shape[1] > 1024:
        transform = aug.get_transform(input_image)
        input_image = transform.apply_image(input_image)
    else:
        transform = None

    start_time = time.time()
    predictions, visualized_output, _, metadata = demo.run_on_image(
        input_image,
        text_prompt=input_text,
        mask_prompt=input_mask,
        with_box=with_box,
        with_mask=with_mask,
        with_sseg=with_sseg,
    )

    logger.info(
        "{} in {:.2f}s".format(
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )

    output_image = visualized_output.get_image()
    print("output_image", output_image.shape)
    # if input_format == "RGB":
    #     output_image = output_image[:, :, ::-1]
    if transform:
        output_image = transform.inverse().apply_image(output_image)
    print("output_image", output_image.shape)

    output_image = Image.fromarray(output_image)

    gc.collect()
    torch.cuda.empty_cache()

    json_results = instances_to_coco_json(predictions["instances"].to(demo.cpu_device), 0)
    for json_result in json_results:
        json_result["category_name"] = metadata.thing_classes[json_result["category_id"]]
        del json_result["image_id"]

    return output_image, json_results


def load_APE_A():
    # init_checkpoint= "output2/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VG/ape_deta/ape_deta_vitl_eva02_lsj_cp_720k_20230504_002019/model_final.pth"
    init_checkpoint = "configs/LVISCOCOCOCOSTUFF_O365_OID_VG/ape_deta/ape_deta_vitl_eva02_lsj_cp_720k_20230504_002019/model_final.pth"
    init_checkpoint = hf_hub_download(repo_id=ckpt_repo_id, filename=init_checkpoint)

    args = get_parser().parse_args()
    args.config_file = get_config_file(
        "LVISCOCOCOCOSTUFF_O365_OID_VG/ape_deta/ape_deta_vitl_eva02_lsj1024_cp_720k.py"
    )
    args.confidence_threshold = 0.01
    args.opts = [
        "train.init_checkpoint='{}'".format(init_checkpoint),
        "model.model_language.cache_dir=''",
        "model.model_vision.select_box_nums_for_evaluation=500",
        "model.model_vision.backbone.net.xattn=False",
        "model.model_vision.transformer.encoder.pytorch_attn=True",
        "model.model_vision.transformer.decoder.pytorch_attn=True",
    ]
    if running_device == "cpu":
        args.opts += [
            "model.model_language.dtype='float32'",
        ]
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    cfg.model.model_vision.criterion[0].use_fed_loss = False
    cfg.model.model_vision.criterion[2].use_fed_loss = False
    cfg.train.device = running_device

    ape.modeling.text.eva01_clip.eva_clip._MODEL_CONFIGS[cfg.model.model_language.clip_model][
        "vision_cfg"
    ]["layers"] = 1
    ape.modeling.text.eva01_clip.eva_clip._MODEL_CONFIGS[cfg.model.model_language.clip_model][
        "vision_cfg"
    ]["fusedLN"] = False

    demo = VisualizationDemo(cfg, args=args)
    if save_memory:
        demo.predictor.model.to("cpu")
        # demo.predictor.model.half()
    else:
        demo.predictor.model.to(running_device)

    all_demo["APE_A"] = demo
    all_cfg["APE_A"] = cfg


def load_APE_B():
    # init_checkpoint= "output2/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj_cp_1080k_20230702_225418/model_final.pth"
    init_checkpoint = "configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj_cp_1080k_20230702_225418/model_final.pth"
    init_checkpoint = hf_hub_download(repo_id=ckpt_repo_id, filename=init_checkpoint)

    args = get_parser().parse_args()
    args.config_file = get_config_file(
        "LVISCOCOCOCOSTUFF_O365_OID_VGR_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_cp_1080k.py"
    )
    args.confidence_threshold = 0.01
    args.opts = [
        "train.init_checkpoint='{}'".format(init_checkpoint),
        "model.model_language.cache_dir=''",
        "model.model_vision.select_box_nums_for_evaluation=500",
        "model.model_vision.text_feature_bank_reset=True",
        "model.model_vision.backbone.net.xattn=False",
        "model.model_vision.transformer.encoder.pytorch_attn=True",
        "model.model_vision.transformer.decoder.pytorch_attn=True",
    ]
    if running_device == "cpu":
        args.opts += [
            "model.model_language.dtype='float32'",
        ]
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    cfg.model.model_vision.criterion[0].use_fed_loss = False
    cfg.model.model_vision.criterion[2].use_fed_loss = False
    cfg.train.device = running_device

    ape.modeling.text.eva01_clip.eva_clip._MODEL_CONFIGS[cfg.model.model_language.clip_model][
        "vision_cfg"
    ]["layers"] = 1
    ape.modeling.text.eva01_clip.eva_clip._MODEL_CONFIGS[cfg.model.model_language.clip_model][
        "vision_cfg"
    ]["fusedLN"] = False

    demo = VisualizationDemo(cfg, args=args)
    if save_memory:
        demo.predictor.model.to("cpu")
        # demo.predictor.model.half()
    else:
        demo.predictor.model.to(running_device)

    all_demo["APE_B"] = demo
    all_cfg["APE_B"] = cfg


def load_APE_C():
    # init_checkpoint= "output2/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj_cp_1080k_20230702_210950/model_final.pth"
    init_checkpoint = "configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj_cp_1080k_20230702_210950/model_final.pth"
    init_checkpoint = hf_hub_download(repo_id=ckpt_repo_id, filename=init_checkpoint)

    args = get_parser().parse_args()
    args.config_file = get_config_file(
        "LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_cp_1080k.py"
    )
    args.confidence_threshold = 0.01
    args.opts = [
        "train.init_checkpoint='{}'".format(init_checkpoint),
        "model.model_language.cache_dir=''",
        "model.model_vision.select_box_nums_for_evaluation=500",
        "model.model_vision.text_feature_bank_reset=True",
        "model.model_vision.backbone.net.xattn=False",
        "model.model_vision.transformer.encoder.pytorch_attn=True",
        "model.model_vision.transformer.decoder.pytorch_attn=True",
    ]
    if running_device == "cpu":
        args.opts += [
            "model.model_language.dtype='float32'",
        ]
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    cfg.model.model_vision.criterion[0].use_fed_loss = False
    cfg.model.model_vision.criterion[2].use_fed_loss = False
    cfg.train.device = running_device

    ape.modeling.text.eva01_clip.eva_clip._MODEL_CONFIGS[cfg.model.model_language.clip_model][
        "vision_cfg"
    ]["layers"] = 1
    ape.modeling.text.eva01_clip.eva_clip._MODEL_CONFIGS[cfg.model.model_language.clip_model][
        "vision_cfg"
    ]["fusedLN"] = False

    demo = VisualizationDemo(cfg, args=args)
    if save_memory:
        demo.predictor.model.to("cpu")
        # demo.predictor.model.half()
    else:
        demo.predictor.model.to(running_device)

    all_demo["APE_C"] = demo
    all_cfg["APE_C"] = cfg


def load_APE_D():
    # init_checkpoint= "output2/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl_20230829_162438/model_final.pth"
    init_checkpoint = "configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl_20230829_162438/model_final.pth"
    init_checkpoint = hf_hub_download(repo_id=ckpt_repo_id, filename=init_checkpoint)

    args = get_parser().parse_args()
    args.config_file = get_config_file(
        "LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py"
    )
    args.confidence_threshold = 0.01
    args.opts = [
        "train.init_checkpoint='{}'".format(init_checkpoint),
        "model.model_language.cache_dir=''",
        "model.model_vision.select_box_nums_for_evaluation=500",
        "model.model_vision.text_feature_bank_reset=True",
        "model.model_vision.backbone.net.xattn=False",
        "model.model_vision.transformer.encoder.pytorch_attn=True",
        "model.model_vision.transformer.decoder.pytorch_attn=True",
    ]
    if running_device == "cpu":
        args.opts += [
            "model.model_language.dtype='float32'",
        ]
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    cfg.model.model_vision.criterion[0].use_fed_loss = False
    cfg.model.model_vision.criterion[2].use_fed_loss = False
    cfg.train.device = running_device

    ape.modeling.text.eva02_clip.factory._MODEL_CONFIGS[cfg.model.model_language.clip_model][
        "vision_cfg"
    ]["layers"] = 1

    demo = VisualizationDemo(cfg, args=args)
    if save_memory:
        demo.predictor.model.to("cpu")
        # demo.predictor.model.half()
    else:
        demo.predictor.model.to(running_device)

    all_demo["APE_D"] = demo
    all_cfg["APE_D"] = cfg


def APE_A_tab():
    with gr.Tab("APE A"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                input_image = gr.Image(
                    sources=["upload"],
                    type="filepath",
                    # tool="sketch",
                    # brush_radius=50,
                )
                input_text = gr.Textbox(
                    label="Object Prompt (optional, if not provided, will only find COCO object.)",
                    info="格式: word1,word2,word3,...",
                )

                score_threshold = gr.Slider(
                    label="Score Threshold", minimum=0.01, maximum=1.0, value=0.3, step=0.01
                )

                output_type = gr.CheckboxGroup(
                    ["object detection", "instance segmentation"],
                    value=["object detection", "instance segmentation"],
                    label="Output Type",
                    info="Which kind of output is displayed?",
                ).style(item_container=True, container=True)

                run_button = gr.Button("Run")

            with gr.Column(scale=2):
                gallery = gr.Image(
                    type="pil",
                )

        example_data = gr.Dataset(
            components=[input_image, input_text, score_threshold],
            samples=examples,
            samples_per_page=5,
        )
        example_data.click(fn=set_example, inputs=example_data, outputs=example_data.components)

        # add_tail_info()
        output_json = gr.JSON(label="json results")

        run_button.click(
            fn=run_on_image,
            inputs=[input_image, input_text, score_threshold, output_type],
            outputs=[gallery, output_json],
        )


def APE_C_tab():
    with gr.Tab("APE C"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                input_image = gr.Image(
                    sources=["upload"],
                    type="filepath",
                    # tool="sketch",
                    # brush_radius=50,
                )
                input_text = gr.Textbox(
                    label="Object Prompt (optional, if not provided, will only find COCO object.)",
                    info="格式: word1,word2,sentence1,sentence2,...",
                )

                score_threshold = gr.Slider(
                    label="Score Threshold", minimum=0.01, maximum=1.0, value=0.3, step=0.01
                )

                output_type = gr.CheckboxGroup(
                    ["object detection", "instance segmentation", "semantic segmentation"],
                    value=["object detection", "instance segmentation"],
                    label="Output Type",
                    info="Which kind of output is displayed?",
                ).style(item_container=True, container=True)

                run_button = gr.Button("Run")

            with gr.Column(scale=2):
                gallery = gr.Image(
                    type="pil",
                )

        example_data = gr.Dataset(
            components=[input_image, input_text, score_threshold],
            samples=example_list,
            samples_per_page=5,
        )
        example_data.click(fn=set_example, inputs=example_data, outputs=example_data.components)

        # add_tail_info()
        output_json = gr.JSON(label="json results")

        run_button.click(
            fn=run_on_image_C,
            inputs=[input_image, input_text, score_threshold, output_type],
            outputs=[gallery, output_json],
        )


def APE_D_tab():
    with gr.Tab("APE D"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                input_image = gr.Image(
                    sources=["upload"],
                    type="filepath",
                    # tool="sketch",
                    # brush_radius=50,
                )
                input_text = gr.Textbox(
                    label="Object Prompt (optional, if not provided, will only find COCO object.)",
                    info="格式: word1,word2,sentence1,sentence2,...",
                )

                score_threshold = gr.Slider(
                    label="Score Threshold", minimum=0.01, maximum=1.0, value=0.1, step=0.01
                )

                output_type = gr.CheckboxGroup(
                    ["object detection", "instance segmentation", "semantic segmentation"],
                    value=["object detection", "instance segmentation"],
                    label="Output Type",
                    info="Which kind of output is displayed?",
                )

                run_button = gr.Button("Run")

            with gr.Column(scale=2):
                gallery = gr.Image(
                    type="pil",
                )

        gr.Examples(
            examples=example_list,
            inputs=[input_image, input_text, score_threshold, output_type],
            examples_per_page=20,
        )

        # add_tail_info()
        output_json = gr.JSON(label="json results")

        run_button.click(
            fn=run_on_image_D,
            inputs=[input_image, input_text, score_threshold, output_type],
            outputs=[gallery, output_json],
        )


def comparison_tab():
    with gr.Tab("APE all"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                input_image = gr.Image(
                    sources=["upload"],
                    type="filepath",
                    # tool="sketch",
                    # brush_radius=50,
                )
                input_text = gr.Textbox(
                    label="Object Prompt (optional, if not provided, will only find COCO object.)",
                    info="格式: word1,word2,sentence1,sentence2,...",
                )

                score_threshold = gr.Slider(
                    label="Score Threshold", minimum=0.01, maximum=1.0, value=0.1, step=0.01
                )

                output_type = gr.CheckboxGroup(
                    ["object detection", "instance segmentation", "semantic segmentation"],
                    value=["object detection", "instance segmentation"],
                    label="Output Type",
                    info="Which kind of output is displayed?",
                )

                run_button = gr.Button("Run")

            gallery_all = []
            with gr.Column(scale=2):
                for key in all_demo.keys():
                    gallery = gr.Image(
                        label=key,
                        type="pil",
                    )
                    gallery_all.append(gallery)

        gr.Examples(
            examples=example_list,
            inputs=[input_image, input_text, score_threshold, output_type],
            examples_per_page=20,
        )

        # add_tail_info()

        run_button.click(
            fn=run_on_image_comparison,
            inputs=[input_image, input_text, score_threshold, output_type],
            outputs=gallery_all,
        )


def is_port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def add_head_info(max_available_memory):
    gr.Markdown(
        "# APE: Aligning and Prompting Everything All at Once for Universal Visual Perception"
    )
    if max_available_memory:
        gr.Markdown(
            "Note multiple models are deployed on single GPU, so it may take several minutes to run the models and visualize the results."
        )
    else:
        gr.Markdown(
            "Note multiple models are deployed on CPU, so it may take a while to run the models and visualize the results."
        )
        gr.Markdown(
            "Noted results computed by CPU are slightly different to results computed by GPU, and some libraries are disabled on CPU."
        )
    gr.Markdown(
        "If the demo is out of memory, try to ***decrease*** the number of object prompt and ***increase*** score threshold."
    )

    gr.Markdown("---")


def add_tail_info():
    gr.Markdown("---")
    gr.Markdown("### We also support Prompt")
    gr.Markdown(
        """
    |  Location prompt   | result |  Location prompt   | result  |
    |  ----  | ----  |  ----  | ----  |
    | ![Location prompt](/file=examples/prompt/20230627-131346_11.176.20.67_mask.PNG)  | ![结果](/file=examples/prompt/20230627-131346_11.176.20.67_pred.png) | ![Location prompt](/file=examples/prompt/20230627-131530_11.176.20.67_mask.PNG)  | ![结果](/file=examples/prompt/20230627-131530_11.176.20.67_pred.png) |
    | ![Location prompt](/file=examples/prompt/20230627-131520_11.176.20.67_mask.PNG)  | ![结果](/file=examples/prompt/20230627-131520_11.176.20.67_pred.png) | ![Location prompt](/file=examples/prompt/20230627-114219_11.176.20.67_mask.PNG)  | ![结果](/file=examples/prompt/20230627-114219_11.176.20.67_pred.png) |
    """
    )
    gr.Markdown("---")


if __name__ == "__main__":
    available_port = [80, 8080]
    for port in available_port:
        if is_port_in_use(port):
            continue
        else:
            server_port = port
            break
    print("server_port", server_port)

    available_memory = [
        torch.cuda.mem_get_info(i)[0] / 1024**3 for i in range(torch.cuda.device_count())
    ]

    global running_device
    if len(available_memory) > 0:
        max_available_memory = max(available_memory)
        device_id = available_memory.index(max_available_memory)

        running_device = "cuda:" + str(device_id)
    else:
        max_available_memory = 0
        running_device = "cpu"

    global save_memory
    save_memory = False
    if max_available_memory > 0 and max_available_memory < 40:
        save_memory = True

    print("available_memory", available_memory)
    print("max_available_memory", max_available_memory)
    print("running_device", running_device)
    print("save_memory", save_memory)

    # ==========================================================================================

    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    setup_logger(name="ape")
    global logger
    logger = setup_logger()

    global aug
    aug = T.ResizeShortestEdge([1024, 1024], 1024)

    global all_demo
    all_demo = {}
    all_cfg = {}

    # load_APE_A()
    # load_APE_B()
    # load_APE_C()
    save_memory = False
    load_APE_D()

    title = "APE: Aligning and Prompting Everything All at Once for Universal Visual Perception"
    block = gr.Blocks(title=title).queue()
    with block:
        add_head_info(max_available_memory)

        # APE_A_tab()
        # APE_C_tab()
        APE_D_tab()

        comparison_tab()

        # add_tail_info()

    block.launch(
        share=False,
        # server_name="0.0.0.0",
        # server_port=server_port,
        show_api=False,
        show_error=True,
    )
