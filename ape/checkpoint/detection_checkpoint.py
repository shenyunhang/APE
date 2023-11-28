# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import pickle
from collections import defaultdict
from typing import IO, Any, Dict, Iterable, List, NamedTuple, Optional, Tuple, cast

import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer as DetectionCheckpointer_d2


class DetectionCheckpointer(DetectionCheckpointer_d2):

    # def __init__(self, skip_key="", **kwargs):
    #     super().__init__(**kwargs)
    #     self.skip_key = skip_key

    def _convert_ndarray_to_tensor(self, state_dict: Dict[str, Any]) -> None:
        """
        In-place convert all numpy arrays in the state_dict to torch tensor.
        Args:
            state_dict (dict): a state-dict to be loaded to the model.
                Will be modified.
        """
        logger = logging.getLogger(__name__)
        # model could be an OrderedDict with _metadata attribute
        # (as returned by Pytorch's state_dict()). We should preserve these
        # properties.
        for k in list(state_dict.keys()):

            # if self.skip_key in k:
            # if "model_language" in k:
            #     state_dict.pop(k)
            #     continue

            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
                logger.warning("Unsupported type found in checkpoint! {}: {}".format(k, type(v)))
                state_dict.pop(k)
                continue
                raise ValueError("Unsupported type found in checkpoint! {}: {}".format(k, type(v)))
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)
