# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import json
import multiprocessing as mp
import os
import tempfile
import time
import warnings
from collections import abc

import cv2
import numpy as np
import tqdm

from detectron2.config import LazyConfig, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

# from detectron2.projects.deeplab import add_deeplab_config
# from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2.utils.logger import setup_logger
from predictor_lazy import VisualizationDemo

# constants
WINDOW_NAME = "APE"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    if "output_dir" in cfg.model:
        cfg.model.output_dir = cfg.train.output_dir
    if "model_vision" in cfg.model and "output_dir" in cfg.model.model_vision:
        cfg.model.model_vision.output_dir = cfg.train.output_dir
    if "train" in cfg.dataloader:
        if isinstance(cfg.dataloader.train, abc.MutableSequence):
            for i in range(len(cfg.dataloader.train)):
                if "output_dir" in cfg.dataloader.train[i].mapper:
                    cfg.dataloader.train[i].mapper.output_dir = cfg.train.output_dir
        else:
            if "output_dir" in cfg.dataloader.train.mapper:
                cfg.dataloader.train.mapper.output_dir = cfg.train.output_dir

    if "model_vision" in cfg.model:
        cfg.model.model_vision.test_score_thresh = args.confidence_threshold
    else:
        cfg.model.test_score_thresh = args.confidence_threshold

    # default_setup(cfg, args)

    setup_logger(name="ape")
    setup_logger(name="timm")

    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--text-prompt", default=None)

    parser.add_argument("--with-box", action="store_true", help="show box of instance")
    parser.add_argument("--with-mask", action="store_true", help="show mask of instance")
    parser.add_argument("--with-sseg", action="store_true", help="show mask of class")

    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    setup_logger(name="ape")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    if args.video_input:
        demo = VisualizationDemo(cfg, parallel=True, args=args)
    else:
        demo = VisualizationDemo(cfg, args=args)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]), recursive=True)
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            try:
                img = read_image(path, format="BGR")
            except Exception as e:
                print("*" * 60)
                print("fail to open image: ", e)
                print("*" * 60)
                continue
            start_time = time.time()
            predictions, visualized_output, visualized_outputs, metadata = demo.run_on_image(
                img,
                text_prompt=args.text_prompt,
                with_box=args.with_box,
                with_mask=args.with_mask,
                with_sseg=args.with_sseg,
            )
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                out_filename = out_filename.replace(".webp", ".png")
                out_filename = out_filename.replace(".crdownload", ".png")
                out_filename = out_filename.replace(".jfif", ".png")
                visualized_output.save(out_filename)

                for i in range(len(visualized_outputs)):
                    out_filename = (
                        os.path.join(args.output, os.path.basename(path)) + "." + str(i) + ".png"
                    )
                    visualized_outputs[i].save(out_filename)

                # import pickle
                # with open(out_filename + ".pkl", "wb") as outp:
                #     pickle.dump(predictions, outp, pickle.HIGHEST_PROTOCOL)

                if "instances" in predictions:
                    results = instances_to_coco_json(
                        predictions["instances"].to(demo.cpu_device), path
                    )
                    for result in results:
                        result["category_name"] = metadata.thing_classes[result["category_id"]]
                        result["image_name"] = result["image_id"]

                    with open(out_filename + ".json", "w") as outp:
                        json.dump(results, outp)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        codec, file_ext = "mp4v", ".mp4"
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        # i = 0
        assert os.path.isfile(args.video_input)
        for vis_frame, predictions in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)

                # import pickle
                # with open(output_fname + "." + str(i) + ".pkl", "wb") as outp:
                #     pickle.dump(predictions, outp, pickle.HIGHEST_PROTOCOL)
                # i += 1
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
