# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import pickle
from typing import IO, Any, Dict, Iterable, List, NamedTuple, Optional, Tuple, cast

import numpy as np
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import FullStateDictConfig

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


class FSDPDetectionCheckpointer(DetectionCheckpointer):

    # def __init__(self, skip_key="", **kwargs):
    #     super().__init__(**kwargs)
    #     self.skip_key = skip_key

    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        # if not self.save_dir or not self.save_to_disk:
        #     return

        data = {}

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self.model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            data["model"] = self.model.state_dict()

        if not self.save_dir or not self.save_to_disk:
            return

        # data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            # pyre-fixme[22]: The cast is redundant.
            torch.save(data, cast(IO[bytes], f))
        self.tag_last_checkpoint(basename)

