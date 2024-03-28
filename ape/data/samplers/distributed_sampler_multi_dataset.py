# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import math
from collections import defaultdict
from typing import Optional

import torch
from torch.utils.data.sampler import Sampler

from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.utils import comm

logger = logging.getLogger(__name__)


class MultiDatasetTrainingSampler(Sampler):
    def __init__(self, repeat_factors, *, shuffle=True, seed=None):
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        # Split into whole number (_int_part) and fractional (_frac_part) parts.
        self._int_part = torch.trunc(repeat_factors)
        self._frac_part = repeat_factors - self._int_part

    @staticmethod
    def get_repeat_factors(
        dataset_dicts, num_datasets, dataset_ratio, use_rfs, use_cas, repeat_thresh, cas_lambda
    ):
        sizes = [0 for _ in range(num_datasets)]
        for d in dataset_dicts:
            sizes[d["dataset_id"]] += 1

        assert len(dataset_ratio) == len(
            sizes
        ), "length of dataset ratio {} should be equal to number if dataset {}".format(
            len(dataset_ratio), len(sizes)
        )
        dataset_weight = [
            torch.ones(s, dtype=torch.float32) * max(sizes) / s * r
            for i, (r, s) in enumerate(zip(dataset_ratio, sizes))
        ]

        logger = logging.getLogger(__name__)
        logger.info(
            "Training sampler dataset weight: {}".format(
                str([max(sizes) / s * r for i, (r, s) in enumerate(zip(dataset_ratio, sizes))])
            )
        )

        st = 0
        repeat_factors = []
        for i, s in enumerate(sizes):
            assert use_rfs[i] * use_cas[i] == 0
            if use_rfs[i]:
                repeat_factor = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                    dataset_dicts[st : st + s], repeat_thresh
                )
            elif use_cas[i]:
                repeat_factor = MultiDatasetTrainingSampler.get_class_balance_factor_per_dataset(
                    dataset_dicts[st : st + s], l=cas_lambda
                )
                repeat_factor = repeat_factor * (s / repeat_factor.sum())
            else:
                repeat_factor = torch.ones(s, dtype=torch.float32)
            logger.info(
                "Training sampler class weight: {} {} {}".format(
                    repeat_factor.size(), repeat_factor.max(), repeat_factor.min()
                )
            )
            repeat_factors.append(repeat_factor)
            st = st + s
        repeat_factors = torch.cat(repeat_factors)
        dataset_weight = torch.cat(dataset_weight)
        repeat_factors = dataset_weight * repeat_factors

        return repeat_factors

    @staticmethod
    def get_class_balance_factor_per_dataset(dataset_dicts, l=1.0):
        rep_factors = []
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            rep_factor = sum([1.0 / (category_freq[cat_id] ** l) for cat_id in cat_ids])
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.

        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            indices = self._get_epoch_indices(g)
            if self._shuffle:
                randperm = torch.randperm(len(indices), generator=g)
                yield from indices[randperm].tolist()
            else:
                yield from indices.tolist()


class InferenceSampler(Sampler):
    """
    Produce indices for inference across all workers.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        if end - begin < max(shard_sizes):
            assert begin > 0
            begin = begin - 1
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
