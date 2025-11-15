import functools
from pathlib import Path
from typing import Any, Optional, Type

import torch
import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule
from torchmetrics.collections import MetricCollection

Optimizer = Type[optim.Optimizer] | functools.partial
Scheduler = Type[optim.lr_scheduler._LRScheduler] | functools.partial


class FrameModule(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        loss: nn.Module,
        optimizer_class: Optimizer,
        metric: Optional[MetricCollection] = None,
        scheduler_config: dict[str, Any] = None,
        cache_entry: str = "",
        compiled: bool = True,
    ):
        super().__init__()
        self.network = torch.compile(network) if compiled else network
        self.loss_fn = loss
        self.metric = metric
        self.optimizer_class = optimizer_class
        self.scheduler_config = scheduler_config
        self.use_cache = cache_entry != ""
        self.cache_entry = cache_entry

        if cache_entry:
            self._forward = self._cached_forward

    def _forward(self, batch, **kwargs):
        images, K, d = batch["images"], batch["intrinsics_norm"]["K"], batch["intrinsics_norm"]["d"]

        return self.network(images, K=K, d=d)

    def _cached_forward(self, batch, **kwargs):
        joints_3D = batch["cache"][self.cache_entry]
        lTm, rTm = batch["transforms"]["egocam_left_to_egocam_middle"], batch["transforms"]["egocam_right_to_egocam_middle"]
        mTw = batch["poses"]["vr"]["egocam_middle"]

        return self.network(joints_3D, middle2world=mTw, left2middle=lTm, right2middle=rTm)

    def training_step(self, batch, batch_idx):
        output = self._forward(batch)

        # Compute loss
        loss, details = self.loss_fn(output, batch)

        self.log_loss(details, prefix="train", batch_size=batch["skeleton"].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        output = self._forward(batch)

        loss, details = self.loss_fn(output, batch)

        self.log_loss(details, prefix="val/loss")

        if self.metric is not None:
            self.metric.update(output, batch)

        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.network.parameters())
        scheduler_config = self.scheduler_config

        if scheduler_config is not None:
            scheduler_class = scheduler_config["scheduler"]

            scheduler_config["scheduler"] = scheduler_class(optimizer)
            return [optimizer], [dict(scheduler_config)]  # To be sure it is not a OmegaConf object

        return optimizer

    def store_hydra_configuration(self, hparams):
        self.save_hyperparameters(dict(hydra=hparams))

    @classmethod
    def load(cls, name: Path | str, **kwargs):
        from framevision.utils.autoloading import load_model

        return load_model(name, **kwargs)

    @classmethod
    def load_from_file(cls, checkpoint_path: Path):
        from framevision.utils.autoloading import load_model_from_file

        return load_model_from_file(checkpoint_path)

    def on_validation_epoch_end(self):
        if self.metric is not None:
            self.log_dict(self.metric.compute(), on_step=False, on_epoch=True)

        if self.metric is not None:
            self.metric.reset()

        return super().on_validation_epoch_end()

    def log_loss(self, loss_details, prefix="", batch_size=None):
        for key, value in loss_details.items():
            self.log(f"{prefix}/{key}", value.mean(), batch_size)
