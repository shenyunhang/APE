# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import gc
import json
import multiprocessing as mp
import time
from collections import deque

import cv2
import numpy as np
import torch

from ape.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


def filter_instances(instances, metadata):
    # return instances

    keep = []
    keep_classes = []

    sorted_idxs = np.argsort(-instances.scores)
    instances = instances[sorted_idxs]

    for i in range(len(instances)):
        instance = instances[i]
        pred_class = instance.pred_classes
        if pred_class >= len(metadata.thing_classes):
            continue

        keep.append(i)
        keep_classes.append(pred_class)
    return instances[keep]


def cuda_grabcut(img, masks, iter=5, gamma=50, iou_threshold=0.75):
    gc.collect()
    torch.cuda.empty_cache()

    try:
        import grabcut
    except Exception as e:
        print("*" * 60)
        print("fail to import grabCut: ", e)
        print("*" * 60)
        return masks
    GC = grabcut.GrabCut(iter)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    tic_0 = time.time()
    for i in range(len(masks)):
        mask = masks[i]
        if mask.sum() > 10 * 10:
            pass
        else:
            continue

        # ----------------------------------------------------------------
        fourmap = np.empty_like(mask, dtype=np.uint8)
        fourmap[:, :] = 64
        fourmap[mask == 0] = 64
        fourmap[mask == 1] = 128

        # Compute segmentation
        tic = time.time()
        seg = GC.estimateSegmentationFromFourmap(img, fourmap, gamma)
        toc = time.time()
        print("Time elapsed in GrabCut segmentation: " + str(toc - tic))
        # ----------------------------------------------------------------

        seg = torch.tensor(seg, dtype=torch.bool)
        iou = (mask & seg).sum() / (mask | seg).sum()
        if iou > iou_threshold:
            masks[i] = seg

        if toc - tic_0 > 10:
            break

    return masks


def opencv_grabcut(img, masks, iter=5):

    for i in range(len(masks)):
        mask = masks[i]

        # ----------------------------------------------------------------
        fourmap = np.empty_like(mask, dtype=np.uint8)
        fourmap[:, :] = cv2.GC_PR_BGD
        # fourmap[mask == 0] = cv2.GC_BGD
        fourmap[mask == 0] = cv2.GC_PR_BGD
        fourmap[mask == 1] = cv2.GC_PR_FGD
        # fourmap[mask == 1] = cv2.GC_FGD

        # Create GrabCut algo
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        seg = np.zeros_like(fourmap, dtype=np.uint8)

        # Compute segmentation
        tic = time.time()
        seg, bgd_model, fgd_model = cv2.grabCut(
            img, fourmap, None, bgd_model, fgd_model, iter, cv2.GC_INIT_WITH_MASK
        )
        toc = time.time()
        print("Time elapsed in GrabCut segmentation: " + str(toc - tic))

        seg = np.where((seg == 2) | (seg == 0), 0, 1).astype("bool")

        # ----------------------------------------------------------------

        seg = torch.tensor(seg, dtype=torch.bool)
        iou = (mask & seg).sum() / (mask | seg).sum()
        if iou > 0.75:
            masks[i] = seg

        if i > 10:
            break

    return masks


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False, args=None):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            "__unused_" + "_".join([d for d in cfg.dataloader.train.dataset.names])
        )
        self.metadata.thing_classes = [
            c
            for d in cfg.dataloader.train.dataset.names
            for c in MetadataCatalog.get(d).get("thing_classes", default=[])
            + MetadataCatalog.get(d).get("stuff_classes", default=["thing"])[1:]
        ]
        self.metadata.stuff_classes = [
            c
            for d in cfg.dataloader.train.dataset.names
            for c in MetadataCatalog.get(d).get("thing_classes", default=[])
            + MetadataCatalog.get(d).get("stuff_classes", default=["thing"])[1:]
        ]

        # self.metadata = MetadataCatalog.get(
        #     "__unused_ape_" + "_".join([d for d in cfg.dataloader.train.dataset.names])
        # )
        # self.metadata.thing_classes = [
        #     c
        #     for d in ["coco_2017_train_panoptic_separated"]
        #     for c in MetadataCatalog.get(d).get("thing_classes", default=[])
        #     + MetadataCatalog.get(d).get("stuff_classes", default=["thing"])[1:]
        # ]
        # self.metadata.stuff_classes = [
        #     c
        #     for d in ["coco_2017_train_panoptic_separated"]
        #     for c in MetadataCatalog.get(d).get("thing_classes", default=[])
        #     + MetadataCatalog.get(d).get("stuff_classes", default=["thing"])[1:]
        # ]

        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

        print(args)

    def run_on_image(
        self,
        image,
        text_prompt=None,
        mask_prompt=None,
        with_box=True,
        with_mask=True,
        with_sseg=True,
    ):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        if text_prompt:
            text_list = [x.strip() for x in text_prompt.split(",")]
            text_list = [x for x in text_list if len(x) > 0]
            metadata = MetadataCatalog.get("__unused_ape_" + text_prompt)
            metadata.thing_classes = text_list
            metadata.stuff_classes = text_list
        else:
            metadata = self.metadata

        vis_output = None
        predictions = self.predictor(image, text_prompt, mask_prompt)

        if "instances" in predictions:
            predictions["instances"] = filter_instances(
                predictions["instances"].to(self.cpu_device), metadata
            )

        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, metadata, instance_mode=self.instance_mode)
        vis_outputs = []
        if "panoptic_seg" in predictions and with_mask and with_sseg:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions and with_sseg:
                # vis_output = visualizer.draw_sem_seg(
                #     predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                # )

                sem_seg = predictions["sem_seg"].to(self.cpu_device)
                # sem_seg = opencv_grabcut(image, sem_seg, iter=10)
                # sem_seg = cuda_grabcut(image, sem_seg > 0.5, iter=5, gamma=10, iou_threshold=0.1)
                sem_seg = torch.cat((sem_seg, torch.ones_like(sem_seg[0:1, ...]) * 0.1), dim=0)
                sem_seg = sem_seg.argmax(dim=0)
                vis_output = visualizer.draw_sem_seg(sem_seg)
            if "instances" in predictions and (with_box or with_mask):
                instances = predictions["instances"].to(self.cpu_device)

                if not with_box:
                    instances.remove("pred_boxes")
                if not with_mask:
                    instances.remove("pred_masks")

                if with_mask and False:
                    # instances.pred_masks = opencv_grabcut(image, instances.pred_masks, iter=10)
                    instances.pred_masks = cuda_grabcut(
                        image, instances.pred_masks, iter=5, gamma=10, iou_threshold=0.75
                    )

                vis_output = visualizer.draw_instance_predictions(predictions=instances)

                # for i in range(len(instances)):
                #     visualizer = Visualizer(image, metadata, instance_mode=self.instance_mode)
                #     vis_outputs.append(visualizer.draw_instance_predictions(predictions=instances[i]))

            elif "proposals" in predictions:
                visualizer = Visualizer(image, None, instance_mode=self.instance_mode)
                instances = predictions["proposals"].to(self.cpu_device)
                instances.pred_boxes = instances.proposal_boxes
                instances.scores = instances.objectness_logits
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output, vis_outputs, metadata

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions and False:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions and False:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions and False:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            if "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
                frame = vis_frame.get_image()

            if "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                predictions = filter_instances(predictions, self.metadata)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame, predictions

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
