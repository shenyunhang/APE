#!/usr/bin/env python
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import random
import sys
import time
from collections import abc
from contextlib import nullcontext
from datetime import timedelta

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

import ape
from ape.checkpoint import DetectionCheckpointer
from ape.engine import SimpleTrainer
from ape.evaluation import inference_on_dataset
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser  # SimpleTrainer,
from detectron2.engine import default_setup, hooks, launch
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import print_csv_format
from detectron2.utils import comm
from detectron2.utils.events import (
    CommonMetricPrinter,
    JSONWriter,
    TensorboardXWriter,
    get_event_storage,
)
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detrex.modeling import ema
from detrex.utils import WandbWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

logger = logging.getLogger("ape")


class Trainer(SimpleTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
        iter_size=1,
        iter_loop=True,
        dataset_ratio=None,
        save_memory=False,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

        self.amp = amp

        self.clip_grad_params = clip_grad_params

        if isinstance(model, DistributedDataParallel):
            if hasattr(model.module, "model_vision"):
                self.dataset_names = model.module.model_vision.dataset_names
            else:
                self.dataset_names = ["unknown"]
        else:
            if hasattr(model, "model_vision"):
                self.dataset_names = model.model_vision.dataset_names
            else:
                self.dataset_names = ["unknown"]
        self.dataset_image_counts = {
            k: torch.tensor(0, dtype=torch.float).to(comm.get_local_rank())
            for k in self.dataset_names
        }
        self.dataset_object_counts = {
            k: torch.tensor(0, dtype=torch.float).to(comm.get_local_rank())
            for k in self.dataset_names
        }

        self.iter_size = iter_size
        self.iter_loop = iter_loop
        self.dataset_ratio = dataset_ratio
        self.save_memory = save_memory

    def run_step(self):
        if self.iter_size > 1:
            if self.iter_loop:
                return self.run_step_accumulate_iter_loop()
            else:
                return self.run_step_accumulate()
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        while True:
            data = next(self._data_loader_iter)
            if all([len(x["instances"]) > 0 for x in data]):
                break
        data_time = time.perf_counter() - start

        for d in data:
            if d.get("dataloader_id", None) is not None:
                d["dataset_id"] = d["dataloader_id"]
            self.dataset_image_counts[self.dataset_names[d.get("dataset_id", 0)]] += 1
            self.dataset_object_counts[self.dataset_names[d.get("dataset_id", 0)]] += len(
                d.get("instances", [])
            )
        dataset_image_counts = {f"count_image/{k}": v for k, v in self.dataset_image_counts.items()}
        dataset_object_counts = {
            f"count_object/{k}": v for k, v in self.dataset_object_counts.items()
        }
        if self.async_write_metrics:
            self.concurrent_executor.submit(
                self._write_metrics_common, dataset_image_counts, iter=self.iter
            )
            self.concurrent_executor.submit(
                self._write_metrics_common, dataset_object_counts, iter=self.iter
            )
        else:
            self._write_metrics_common(dataset_image_counts)
            self._write_metrics_common(dataset_object_counts)

        """
        If you want to do something with the losses, you can wrap the model.
        """
        with autocast(enabled=self.amp):
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

        if self.async_write_metrics:
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        if self.save_memory:
            del losses
            del loss_dict
            torch.cuda.empty_cache()

    def run_step_accumulate(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        while True:
            data = next(self._data_loader_iter)
            if all([len(x["instances"]) > 0 for x in data]):
                break
        data_time = time.perf_counter() - start

        for d in data:
            if d.get("dataloader_id", None) is not None:
                d["dataset_id"] = d["dataloader_id"]
            self.dataset_image_counts[self.dataset_names[d.get("dataset_id", 0)]] += 1
            self.dataset_object_counts[self.dataset_names[d.get("dataset_id", 0)]] += len(
                d.get("instances", [])
            )
        dataset_image_counts = {f"count_image/{k}": v for k, v in self.dataset_image_counts.items()}
        dataset_object_counts = {
            f"count_object/{k}": v for k, v in self.dataset_object_counts.items()
        }
        if self.async_write_metrics:
            self.concurrent_executor.submit(
                self._write_metrics_common, dataset_image_counts, iter=self.iter
            )
            self.concurrent_executor.submit(
                self._write_metrics_common, dataset_object_counts, iter=self.iter
            )
        else:
            self._write_metrics_common(dataset_image_counts)
            self._write_metrics_common(dataset_object_counts)

        sync_context = self.model.no_sync if (self.iter + 1) % self.iter_size != 0 else nullcontext
        """
        If you want to do something with the losses, you can wrap the model.
        """
        with sync_context():
            with autocast(enabled=self.amp):
                loss_dict = self.model(data)

                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        if self.iter == self.start_iter:
            self.optimizer.zero_grad()

        if self.iter_size > 1:
            losses = losses / self.iter_size

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if (self.iter + 1) % self.iter_size == 0:
                if self.clip_grad_params is not None:
                    self.grad_scaler.unscale_(self.optimizer)
                    self.clip_grads(self.model.parameters())
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
        else:
            losses.backward()
            if (self.iter + 1) % self.iter_size == 0:
                if self.clip_grad_params is not None:
                    self.clip_grads(self.model.parameters())
                self.optimizer.step()
                self.optimizer.zero_grad()

        if self.async_write_metrics:
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        if self.save_memory:
            del losses
            del loss_dict
            torch.cuda.empty_cache()

    def run_step_accumulate_iter_loop(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        self.optimizer.zero_grad()
        for inner_iter in range(self.iter_size):
            start = time.perf_counter()
            """
            If you want to do something with the data, you can wrap the dataloader.
            """
            while True:
                data = next(self._data_loader_iter)
                if all([len(x["instances"]) > 0 for x in data]):
                    break
            data_time = time.perf_counter() - start

            for d in data:
                if d.get("dataloader_id", None) is not None:
                    d["dataset_id"] = d["dataloader_id"]
                self.dataset_image_counts[self.dataset_names[d.get("dataset_id", 0)]] += 1
                self.dataset_object_counts[self.dataset_names[d.get("dataset_id", 0)]] += len(
                    d.get("instances", [])
                )
            dataset_image_counts = {
                f"count_image/{k}": v for k, v in self.dataset_image_counts.items()
            }
            dataset_object_counts = {
                f"count_object/{k}": v for k, v in self.dataset_object_counts.items()
            }
            if self.async_write_metrics:
                self.concurrent_executor.submit(
                    self._write_metrics_common, dataset_image_counts, iter=self.iter
                )
                self.concurrent_executor.submit(
                    self._write_metrics_common, dataset_object_counts, iter=self.iter
                )
            else:
                self._write_metrics_common(dataset_image_counts)
                self._write_metrics_common(dataset_object_counts)

            sync_context = self.model.no_sync if inner_iter != self.iter_size - 1 else nullcontext
            """
            If you want to do something with the losses, you can wrap the model.
            """
            with sync_context():
                with autocast(enabled=self.amp):
                    loss_dict = self.model(data)

                    if isinstance(loss_dict, torch.Tensor):
                        losses = loss_dict
                        loss_dict = {"total_loss": loss_dict}
                    else:
                        losses = sum(loss_dict.values())

                """
                If you need to accumulate gradients or do something similar, you can
                wrap the optimizer with your custom `zero_grad()` method.
                """

                losses = losses / self.iter_size

                if self.amp:
                    self.grad_scaler.scale(losses).backward()
                else:
                    losses.backward()

            if self.async_write_metrics:
                self.concurrent_executor.submit(
                    self._write_metrics, loss_dict, data_time, iter=self.iter
                )
            else:
                self._write_metrics(loss_dict, data_time)

            if self.save_memory:
                del losses
                del loss_dict
                torch.cuda.empty_cache()

        if self.amp:
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )

    def state_dict(self):
        ret = super().state_dict()
        if self.grad_scaler and self.amp:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.grad_scaler and self.amp:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])

    @property
    def _data_loader_iter(self):
        if isinstance(self.data_loader, abc.MutableSequence):
            if self._data_loader_iter_obj is None:
                self._data_loader_iter_obj = [iter(x) for x in self.data_loader]
                self._data_loader_indices = []

            if len(self._data_loader_indices) == 0:
                self._data_loader_indices = random.choices(
                    list(range(len(self.data_loader))), weights=self.dataset_ratio, k=10000
                )
            idx = self._data_loader_indices.pop()
            return self._data_loader_iter_obj[idx]

        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self.data_loader)
        return self._data_loader_iter_obj


def do_test(cfg, model, eval_only=False):
    logger = logging.getLogger("ape")
    if "evaluator" in cfg.dataloader:
        if isinstance(model, DistributedDataParallel):
            if hasattr(model.module, "set_eval_dataset"):
                model.module.set_eval_dataset(cfg.dataloader.test.dataset.names)
        else:
            if hasattr(model, "set_eval_dataset"):
                model.set_eval_dataset(cfg.dataloader.test.dataset.names)
        output_dir = os.path.join(
            cfg.train.output_dir, "inference_{}".format(cfg.dataloader.test.dataset.names)
        )
        if "cityscapes" in cfg.dataloader.test.dataset.names:
            pass
        else:
            if isinstance(cfg.dataloader.evaluator, abc.MutableSequence):
                for evaluator in cfg.dataloader.evaluator:
                    evaluator.output_dir = output_dir
            else:
                cfg.dataloader.evaluator.output_dir = output_dir

        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        logger.info(
            "Evaluation results for {} in csv format:".format(cfg.dataloader.test.dataset.names)
        )
        print_csv_format(ret)
        ret = {f"{k}_{cfg.dataloader.test.dataset.names}": v for k, v in ret.items()}
    else:
        ret = {}

    if "evaluators" in cfg.dataloader:
        for test, evaluator in zip(cfg.dataloader.tests, cfg.dataloader.evaluators):
            if isinstance(model, DistributedDataParallel):
                model.module.set_eval_dataset(test.dataset.names)
            else:
                model.set_eval_dataset(test.dataset.names)
            output_dir = os.path.join(
                cfg.train.output_dir, "inference_{}".format(test.dataset.names)
            )
            if isinstance(evaluator, abc.MutableSequence):
                for eva in evaluator:
                    eva.output_dir = output_dir
            else:
                evaluator.output_dir = output_dir
            ret_ = inference_on_dataset(model, instantiate(test), instantiate(evaluator))
            logger.info("Evaluation results for {} in csv format:".format(test.dataset.names))
            print_csv_format(ret_)
            ret.update({f"{k}_{test.dataset.names}": v for k, v in ret_.items()})

    bbox_odinw_AP = {"AP": [], "AP50": [], "AP75": [], "APs": [], "APm": [], "APl": []}
    segm_seginw_AP = {"AP": [], "AP50": [], "AP75": [], "APs": [], "APm": [], "APl": []}
    bbox_rf100_AP = {"AP": [], "AP50": [], "AP75": [], "APs": [], "APm": [], "APl": []}
    for k, v in ret.items():
        for kk, vv in v.items():
            if k.startswith("bbox_odinw") and kk in bbox_odinw_AP and vv == vv:
                bbox_odinw_AP[kk].append(vv)
            if k.startswith("segm_seginw") and kk in segm_seginw_AP and vv == vv:
                segm_seginw_AP[kk].append(vv)
            if k.startswith("bbox_rf100") and kk in bbox_rf100_AP and vv == vv:
                bbox_rf100_AP[kk].append(vv)

    from statistics import median, mean

    logger.info("Evaluation results: {}".format(ret))
    for k, v in bbox_odinw_AP.items():
        if len(v) > 0:
            logger.info(
                "Evaluation results for odinw bbox {}: mean {} median {}".format(
                    k, mean(v), median(v)
                )
            )
    for k, v in segm_seginw_AP.items():
        if len(v) > 0:
            logger.info(
                "Evaluation results for seginw segm {}: mean {} median {}".format(
                    k, mean(v), median(v)
                )
            )
    for k, v in bbox_rf100_AP.items():
        if len(v) > 0:
            logger.info(
                "Evaluation results for rf100 bbox {}: mean {} median {}".format(
                    k, mean(v), median(v)
                )
            )

    return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("ape")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    if "wait_group" in cfg.dataloader:
        wait = comm.get_local_rank() % cfg.dataloader.wait_group * cfg.dataloader.wait_time
        logger.info("rank {} sleep {}".format(comm.get_local_rank(), wait))
        time.sleep(wait)
    if isinstance(cfg.dataloader.train, abc.MutableSequence):
        train_loader = [instantiate(x) for x in cfg.dataloader.train]
    else:
        train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)

    ema.may_build_model_ema(cfg, model)

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
        iter_size=cfg.train.iter_size if "iter_size" in cfg.train else 1,
        iter_loop=cfg.train.iter_loop if "iter_loop" in cfg.train else True,
        dataset_ratio=cfg.train.dataset_ratio if "dataset_ratio" in cfg.train else None,
    )

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
        **ema.may_get_ema_checkpointer(cfg, model),
    )

    if comm.is_main_process():
        output_dir = cfg.train.output_dir
        PathManager.mkdirs(output_dir)
        writers = [
            CommonMetricPrinter(cfg.train.max_iter),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
            TensorboardXWriter(output_dir),
        ]
        if cfg.train.wandb.enabled:
            PathManager.mkdirs(cfg.train.wandb.params.dir)
            writers.append(WandbWriter(cfg))

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            ema.EMAHook(cfg, model) if cfg.train.model_ema.enabled else None,
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                writers,
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
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

    default_setup(cfg, args)

    setup_logger(cfg.train.output_dir, distributed_rank=comm.get_rank(), name="ape")
    setup_logger(cfg.train.output_dir, distributed_rank=comm.get_rank(), name="timm")

    if cfg.train.fast_dev_run.enabled:
        cfg.train.max_iter = 20
        cfg.train.eval_period = 10
        cfg.train.log_period = 1

    if args.eval_only:
        model = instantiate(cfg.model)
        logger = logging.getLogger("ape")
        logger.info("Model:\n{}".format(model))
        model.to(cfg.train.device)
        model.to(torch.float16)
        model = create_ddp_model(model)

        ema.may_build_model_ema(cfg, model)
        DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(
            cfg.train.init_checkpoint
        )
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            ema.apply_model_ema(model)
        print(do_test(cfg, model, eval_only=True))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
        timeout=timedelta(minutes=120),
    )
