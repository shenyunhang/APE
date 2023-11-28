# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import operator
import time
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import torch
import torch.utils.data as torchdata
from termcolor import colored
from torch.utils.data.sampler import Sampler

from detectron2.config import configurable
from detectron2.data.build import (
    filter_images_with_few_keypoints,
    filter_images_with_only_crowd_annotations,
    get_detection_dataset_dicts,
    load_proposals_into_dataset,
    trivial_batch_collator,
    worker_init_reset_seed,
)
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import DatasetFromList, MapDataset, ToIterableDataset
from detectron2.data.detection_utils import check_metadata_consistency
from detectron2.data.samplers import (
    RandomSubsetTrainingSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.utils import comm
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import _log_api_usage, log_first_n
from tabulate import tabulate

from .dataset_mapper import DatasetMapper_ape
from .samplers import MultiDatasetTrainingSampler

"""
This file contains the default logic to build a dataloader for training or testing.
"""

__all__ = [
    "build_detection_train_loader_multi_dataset",
]


def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    total_num_out_of_class = 0
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = np.asarray(
            [x["category_id"] for x in annos if not x.get("iscrowd", 0)], dtype=np.int
        )
        if len(classes):
            assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
            # assert (
            #     classes.max() < num_classes
            # ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
        histogram += np.histogram(classes, bins=hist_bins)[0]

        total_num_out_of_class += sum(classes >= num_classes)

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    if total_num_out_of_class > 0:
        data.extend(["total out", total_num_out_of_class])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    log_first_n(
        logging.INFO,
        "Distribution of instances among all {} categories:\n".format(num_classes)
        + colored(table, "cyan"),
        key="message",
    )


def DatasetCatalog_get(dataset_name, reduce_memory, reduce_memory_size):
    import os, psutil

    logger = logging.getLogger(__name__)
    logger.info(
        "Current memory usage: {} GB".format(
            psutil.Process(os.getpid()).memory_info().rss / 1024**3
        )
    )

    dataset_dicts = DatasetCatalog.get(dataset_name)

    # logger.info(
    #     "Current memory usage: {} GB".format(
    #         psutil.Process(os.getpid()).memory_info().rss / 1024**3
    #     )
    # )
    # logger.info("Reducing memory usage...")

    # for d in dataset_dicts:
    #     # LVIS
    #     if "not_exhaustive_category_ids" in d.keys():
    #         del d["not_exhaustive_category_ids"]
    #     if "neg_category_ids" in d.keys():
    #         del d["neg_category_ids"]
    #     if "pos_category_ids" in d.keys():
    #         del d["pos_category_ids"]

    #     if "annotations" not in d.keys():
    #         continue
    #     for anno in d["annotations"]:
    #         if "iscrowd" in anno.keys():
    #             if anno["iscrowd"] == 0:
    #                 del anno["iscrowd"]

    logger.info(
        "Current memory usage: {} GB".format(
            psutil.Process(os.getpid()).memory_info().rss / 1024**3
        )
    )

    if not reduce_memory:
        return dataset_dicts
    if len(dataset_dicts) < reduce_memory_size:
        return dataset_dicts

    logger.info("Reducing memory usage further...")

    for d in dataset_dicts:
        if "annotations" not in d.keys():
            continue

        for anno in d["annotations"]:

            if "bbox" in anno.keys():
                del anno["bbox"]

            if "bbox_mode" in anno.keys():
                del anno["bbox_mode"]

            if "segmentation" in anno.keys():
                del anno["segmentation"]

            if "phrase" in anno.keys():
                del anno["phrase"]

    logger.info(
        "Current memory usage: {} GB".format(
            psutil.Process(os.getpid()).memory_info().rss / 1024**3
        )
    )

    return dataset_dicts


def get_detection_dataset_dicts_multi_dataset(
    names,
    filter_empty=True,
    min_keypoints=0,
    proposal_files=None,
    check_consistency=True,
    filter_emptys=[True],
    dataloader_id=None,
    reduce_memory=False,
    reduce_memory_size=1e6,
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.
        check_consistency (bool): whether to check if datasets have consistent metadata.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    # dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]
    dataset_dicts = [
        DatasetCatalog_get(dataset_name, reduce_memory, reduce_memory_size)
        for dataset_name in names
    ]

    if isinstance(dataset_dicts[0], torchdata.Dataset):
        if len(dataset_dicts) > 1:
            # ConcatDataset does not work for iterable style dataset.
            # We could support concat for iterable as well, but it's often
            # not a good idea to concat iterables anyway.
            return torchdata.ConcatDataset(dataset_dicts)
        return dataset_dicts[0]

    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    for dataset_id, (dataset_name, dicts) in enumerate(zip(names, dataset_dicts)):
        for d in dicts:
            d["dataset_id"] = dataset_id
            if dataloader_id is not None:
                d["dataloader_id"] = dataloader_id

        has_instances = "annotations" in dicts[0]
        if not check_consistency or not has_instances:
            continue
        try:
            class_names = MetadataCatalog.get(dataset_name).thing_classes
            check_metadata_consistency("thing_classes", [dataset_name])
            print_instances_class_histogram(dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass

    assert proposal_files is None
    if proposal_files is not None:
        assert len(names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = [
        filter_images_with_only_crowd_annotations(dicts)
        if flag and "annotations" in dicts[0]
        else dicts
        for dicts, flag in zip(dataset_dicts, filter_emptys)
    ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances and False:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    if check_consistency and has_instances and False:
        try:
            class_names = MetadataCatalog.get(names[0]).thing_classes
            check_metadata_consistency("thing_classes", names)
            print_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    return dataset_dicts


def build_batch_data_loader_multi_dataset(
    dataset,
    sampler,
    total_batch_size,
    total_batch_size_list,
    *,
    aspect_ratio_grouping=False,
    num_workers=0,
    collate_fn=None,
    num_datasets=1,
):
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): a pytorch map-style or iterable dataset.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces indices.
            Must be provided iff. ``dataset`` is a map-style dataset.
        total_batch_size, aspect_ratio_grouping, num_workers, collate_fn: see
            :func:`build_detection_train_loader`.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )
    batch_size = total_batch_size // world_size

    if len(total_batch_size_list) < num_datasets:
        total_batch_size_list += [
            total_batch_size,
        ] * (num_datasets - len(total_batch_size_list))
    assert all([x > 0 for x in total_batch_size_list]) and all(
        [x % world_size == 0 for x in total_batch_size_list]
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_list, world_size
    )
    batch_size = [x // world_size for x in total_batch_size_list]

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        dataset = ToIterableDataset(dataset, sampler)

    assert aspect_ratio_grouping
    if aspect_ratio_grouping:
        data_loader = torchdata.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        # data_loader = AspectRatioGroupedDataset(data_loader, batch_size)
        data_loader = MultiDatasetAspectRatioGroupedDataset(
            data_loader, batch_size, num_datasets=num_datasets
        )
        if collate_fn is None:
            return data_loader
        return MapDataset(data_loader, collate_fn)
    else:
        return torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
            worker_init_fn=worker_init_reset_seed,
        )


def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    assert len(cfg.DATASETS.TRAIN) == len(cfg.MULTI_DATASET.NAMES)
    assert len(cfg.DATASETS.TRAIN) == len(cfg.MULTI_DATASET.ENTITIES)
    assert len(cfg.DATASETS.TRAIN) == len(cfg.MULTI_DATASET.NUM_CLASSES)
    assert len(cfg.DATASETS.TRAIN) == len(cfg.MULTI_DATASET.RATIOS)
    assert len(cfg.DATASETS.TRAIN) == len(cfg.MULTI_DATASET.USE_CAS)
    assert len(cfg.DATASETS.TRAIN) == len(cfg.MULTI_DATASET.USE_RFS)
    assert len(cfg.DATASETS.TRAIN) == len(cfg.MULTI_DATASET.FILTER_EMPTY_ANNOTATIONS)
    # assert len(cfg.DATASETS.TRAIN) == len(cfg.SOLVER.IMS_PER_BATCH_LIST)
    # assert len(cfg.DATASETS.TRAIN) == len(cfg.SOLVER.AUGMENT_TYPE)

    seed1 = comm.shared_random_seed()
    seed2 = comm.shared_random_seed()
    logger = logging.getLogger(__name__)
    logger.info("rank {} seed1 {} seed2 {}".format(comm.get_local_rank(), seed1, seed2))

    # Hard-coded 2 sequent group and 1200s time wait.
    wait_group = 2
    wait_time = cfg.DATALOADER.GROUP_WAIT
    wait = comm.get_local_rank() % wait_group * wait_time
    logger.info("rank {} _train_loader_from_config sleep {}".format(comm.get_local_rank(), wait))
    time.sleep(wait)

    if dataset is None:
        dataset = get_detection_dataset_dicts_multi_dataset(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            filter_emptys=cfg.MULTI_DATASET.FILTER_EMPTY_ANNOTATIONS,
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    if mapper is None:
        mapper = DatasetMapper_ape(cfg, True)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        if isinstance(dataset, torchdata.IterableDataset):
            logger.info("Not using any sampler since the dataset is IterableDataset.")
            sampler = None
        else:
            logger.info("Using training sampler {}".format(sampler_name))
            if sampler_name == "TrainingSampler":
                sampler = TrainingSampler(len(dataset), seed=seed1)
            elif sampler_name == "RepeatFactorTrainingSampler":
                repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                    dataset, cfg.DATALOADER.REPEAT_THRESHOLD
                )
                sampler = RepeatFactorTrainingSampler(repeat_factors, seed=seed1)
            elif sampler_name == "RandomSubsetTrainingSampler":
                sampler = RandomSubsetTrainingSampler(
                    len(dataset),
                    cfg.DATALOADER.RANDOM_SUBSET_RATIO,
                    seed_shuffle=seed1,
                    seed_subset=seed2,
                )
            elif sampler_name == "MultiDatasetSampler":
                raise ValueError("Despreted training sampler: {}".format(sampler_name))
                sizes = [0 for _ in range(len(cfg.DATASETS.TRAIN))]
                for d in dataset:
                    sizes[d["dataset_id"]] += 1
                sampler = MultiDatasetSampler(cfg, dataset, sizes, seed=seed1)
            elif sampler_name == "MultiDatasetTrainingSampler":
                # sampler = MultiDatasetTrainingSampler(cfg, dataset, seed=seed1)
                repeat_factors = MultiDatasetTrainingSampler.get_repeat_factors(
                    dataset,
                    len(cfg.DATASETS.TRAIN),
                    cfg.MULTI_DATASET.RATIOS,
                    cfg.MULTI_DATASET.USE_RFS,
                    cfg.MULTI_DATASET.USE_CAS,
                    cfg.MULTI_DATASET.REPEAT_THRESHOLD,
                    cfg.MULTI_DATASET.CAS_LAMBDA,
                )
                sampler = MultiDatasetTrainingSampler(repeat_factors, seed=seed1)
            else:
                raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "total_batch_size_list": cfg.SOLVER.IMS_PER_BATCH_LIST,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "num_datasets": len(cfg.DATASETS.TRAIN),
    }


@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader_multi_dataset(
    dataset,
    *,
    mapper,
    sampler=None,
    total_batch_size,
    total_batch_size_list,
    aspect_ratio_grouping=True,
    num_workers=0,
    collate_fn=None,
    num_datasets=1,
):
    """
    Build a dataloader for object detection with some default features.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). It can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``.
            If ``dataset`` is map-style, the default sampler is a :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
            Sampler must be None if ``dataset`` is iterable.
        total_batch_size (int): total batch size across all workers.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers
        collate_fn: a function that determines how to do batching, same as the argument of
            `torch.utils.data.DataLoader`. Defaults to do no collation and return a list of
            data. No collation is OK for small batch size and simple data structures.
            If your batch size is large and each sample contains too many small tensors,
            it's more efficient to collate them in data loader.

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    # wait = round(comm.get_local_rank() * 1.0 * len(dataset) / 60000)
    # logger = logging.getLogger(__name__)
    # logger.info("get_detection_dataset_dicts_multi_dataset sleep {}".format(wait))
    # time.sleep(wait)

    if isinstance(sampler, Callable):
        sampler = sampler(dataset)

    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = TrainingSampler(len(dataset))
        assert isinstance(sampler, torchdata.Sampler), f"Expect a Sampler but got {type(sampler)}"
    return build_batch_data_loader_multi_dataset(
        dataset,
        sampler,
        total_batch_size,
        total_batch_size_list,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
        num_datasets=num_datasets,
    )


class MultiDatasetSampler(Sampler):
    def __init__(self, cfg, dataset_dicts, sizes, seed: Optional[int] = None):
        self.sizes = sizes
        self.sample_epoch_size = cfg.MULTI_DATASET.SAMPLE_EPOCH_SIZE
        assert self.sample_epoch_size % cfg.SOLVER.IMS_PER_BATCH == 0, (
            self.sample_epoch_size % cfg.SOLVER.IMS_PER_BATCH == 0
        )
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        dataset_ratio = cfg.MULTI_DATASET.RATIOS
        assert len(dataset_ratio) == len(
            sizes
        ), "length of dataset ratio {} should be equal to number if dataset {}".format(
            len(dataset_ratio), len(sizes)
        )
        dataset_weight = [
            torch.ones(s) * max(sizes) / s * r / sum(dataset_ratio)
            for i, (r, s) in enumerate(zip(dataset_ratio, sizes))
        ]
        st = 0
        cas_factors = []
        for i, s in enumerate(sizes):
            if cfg.MULTI_DATASET.USE_CAS[i]:
                cas_factor = self._get_class_balance_factor_per_dataset(
                    dataset_dicts[st : st + s], l=cfg.MULTI_DATASET.CAS_LAMBDA
                )
                cas_factor = cas_factor * (s / cas_factor.sum())
            else:
                cas_factor = torch.ones(s)
            cas_factors.append(cas_factor)
            st = st + s
        cas_factors = torch.cat(cas_factors)
        dataset_weight = torch.cat(dataset_weight)
        self.weights = dataset_weight * cas_factors

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            ids = torch.multinomial(
                self.weights, self.sample_epoch_size, generator=g, replacement=True
            )
            yield from ids

    def _get_class_balance_factor_per_dataset(self, dataset_dicts, l=1.0):
        ret = []
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            ret.append(sum([1.0 / (category_freq[cat_id] ** l) for cat_id in cat_ids]))
        return torch.tensor(ret).float()


# class MultiDatasetTrainingSampler(Sampler):
#     def __init__(self, cfg, dataset_dicts, *, shuffle=True, seed=None):
#         sizes = [0 for _ in range(len(cfg.DATASETS.TRAIN))]
#         for d in dataset_dicts:
#             sizes[d["dataset_id"]] += 1

#         dataset_ratio = cfg.MULTI_DATASET.RATIOS
#         assert len(dataset_ratio) == len(
#             sizes
#         ), "length of dataset ratio {} should be equal to number if dataset {}".format(
#             len(dataset_ratio), len(sizes)
#         )
#         dataset_weight = [
#             torch.ones(s) * max(sizes) / s * r for i, (r, s) in enumerate(zip(dataset_ratio, sizes))
#         ]

#         logger = logging.getLogger(__name__)
#         logger.info(
#             "Training sampler dataset weight: {}".format(
#                 str([max(sizes) / s * r for i, (r, s) in enumerate(zip(dataset_ratio, sizes))])
#             )
#         )

#         st = 0
#         repeat_factors = []
#         for i, s in enumerate(sizes):
#             assert cfg.MULTI_DATASET.USE_RFS[i] * cfg.MULTI_DATASET.USE_CAS[i] == 0
#             if cfg.MULTI_DATASET.USE_RFS[i]:
#                 repeat_factor = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
#                     dataset_dicts[st : st + s], cfg.MULTI_DATASET.REPEAT_THRESHOLD
#                 )
#             elif cfg.MULTI_DATASET.USE_CAS[i]:
#                 repeat_factor = MultiDatasetTrainingSampler.get_class_balance_factor_per_dataset(
#                     dataset_dicts[st : st + s], l=cfg.MULTI_DATASET.CAS_LAMBDA
#                 )
#                 repeat_factor = repeat_factor * (s / repeat_factor.sum())
#             else:
#                 repeat_factor = torch.ones(s)
#             repeat_factors.append(repeat_factor)
#             st = st + s
#         repeat_factors = torch.cat(repeat_factors)
#         dataset_weight = torch.cat(dataset_weight)
#         repeat_factors = dataset_weight * repeat_factors

#         self._shuffle = shuffle
#         if seed is None:
#             seed = comm.shared_random_seed()
#         self._seed = int(seed)

#         self._rank = comm.get_rank()
#         self._world_size = comm.get_world_size()

#         # Split into whole number (_int_part) and fractional (_frac_part) parts.
#         self._int_part = torch.trunc(repeat_factors)
#         self._frac_part = repeat_factors - self._int_part

#     @staticmethod
#     def get_class_balance_factor_per_dataset(dataset_dicts, l=1.0):
#         rep_factors = []
#         category_freq = defaultdict(int)
#         for dataset_dict in dataset_dicts:  # For each image (without repeats)
#             cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
#             for cat_id in cat_ids:
#                 category_freq[cat_id] += 1
#         for dataset_dict in dataset_dicts:
#             cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
#             rep_factor = sum([1.0 / (category_freq[cat_id] ** l) for cat_id in cat_ids])
#             rep_factors.append(rep_factor)

#         return torch.tensor(rep_factors, dtype=torch.float32)

#     def _get_epoch_indices(self, generator):
#         """
#         Create a list of dataset indices (with repeats) to use for one epoch.

#         Args:
#             generator (torch.Generator): pseudo random number generator used for
#                 stochastic rounding.

#         Returns:
#             torch.Tensor: list of dataset indices to use in one epoch. Each index
#                 is repeated based on its calculated repeat factor.
#         """
#         # Since repeat factors are fractional, we use stochastic rounding so
#         # that the target repeat factor is achieved in expectation over the
#         # course of training
#         rands = torch.rand(len(self._frac_part), generator=generator)
#         rep_factors = self._int_part + (rands < self._frac_part).float()
#         # Construct a list of indices in which we repeat images as specified
#         indices = []
#         for dataset_index, rep_factor in enumerate(rep_factors):
#             indices.extend([dataset_index] * int(rep_factor.item()))
#         return torch.tensor(indices, dtype=torch.int64)

#     def __iter__(self):
#         start = self._rank
#         yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

#     def _infinite_indices(self):
#         g = torch.Generator()
#         g.manual_seed(self._seed)
#         while True:
#             # Sample indices with repeats determined by stochastic rounding; each
#             # "epoch" may have a slightly different size due to the rounding.
#             indices = self._get_epoch_indices(g)
#             if self._shuffle:
#                 randperm = torch.randperm(len(indices), generator=g)
#                 yield from indices[randperm].tolist()
#             else:
#                 yield from indices.tolist()


class MultiDatasetAspectRatioGroupedDataset(torch.utils.data.IterableDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size, num_datasets):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2 * num_datasets)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        for d in self.dataset:
            w, h = d["width"], d["height"]
            bucket_id = 0 if w > h else 1
            bucket_id = d["dataset_id"] * 2 + bucket_id
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_size[d["dataset_id"]]:
                data = bucket[:]
                # Clear bucket first, because code after yield is not
                # guaranteed to execute
                del bucket[:]
                yield data
