# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import random

import numpy as np
import torch.utils.data as data

from detectron2.data.common import _MapIterableDataset
from detectron2.utils.serialize import PicklableWrapper

__all__ = ["MapDataset_coppaste"]


class MapDataset_coppaste(data.Dataset):
    """
    Map a function over the elements in a dataset.
    """

    def __init__(self, dataset, map_func, dataset_bg, sampler_bg):
        """
        Args:
            dataset: a dataset where map function is applied. Can be either
                map-style or iterable dataset. When given an iterable dataset,
                the returned object will also be an iterable dataset.
            map_func: a callable which maps the element in dataset. map_func can
                return None to skip the data (e.g. in case of errors).
                How None is handled depends on the style of `dataset`.
                If `dataset` is map-style, it randomly tries other elements.
                If `dataset` is iterable, it skips the data and tries the next.
        """
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

        self._dataset_bg = dataset_bg
        self._sampler_bg = sampler_bg
        self._sampler_bg_iter = None

    def __new__(cls, dataset, map_func, dataset_bg, sampler_bg):
        is_iterable = isinstance(dataset, data.IterableDataset)
        if is_iterable:
            assert 0
            return _MapIterableDataset(dataset, map_func)
        else:
            return super().__new__(cls)

    def __getnewargs__(self):
        return self._dataset, self._map_func, self._dataset_bg, self._sampler_bg

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        if self._sampler_bg_iter:
            pass
        else:
            self._sampler_bg._seed = np.random.randint(2**31)
            self._sampler_bg_iter = iter(self._sampler_bg)

        while True:
            cur_idx_bg = next(self._sampler_bg_iter)
            data = self._map_func(self._dataset[cur_idx], self._dataset_bg[cur_idx_bg])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )
