import math
import os
from typing import Any, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.utils as vutils
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torchmetrics.classification.accuracy import Accuracy


class LitMLP(pl.LightningModule):
    def __init__(
        self, input_dim: int = 784, embed_dim: int = 128, num_classes: int = 10
    ) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.layer = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes),
        )

        self.predict_step_outputs = []
        self.example_input_array = torch.randn(32, input_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        return self.layer(x)

    def _shared_step(self, x: Tensor) -> Tensor:
        y_hat = self(x)
        return y_hat

    def training_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self._shared_step(x)
        loss = self.loss_fn(y_hat, y)
        self.train_acc(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(
        self, batch: Tensor, batch_idx: int
    ) -> Optional[STEP_OUTPUT]:
        x, y = batch
        y_hat = self._shared_step(x)
        loss = self.loss_fn(y_hat, y)
        self.val_acc(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True)

    def test_step(
        self, batch: Tensor, batch_idx: int
    ) -> Optional[STEP_OUTPUT]:
        x, y = batch
        y_hat = self._shared_step(x)
        loss = self.loss_fn(y_hat, y)
        self.test_acc(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc, prog_bar=True, on_epoch=True)

    def predict_step(
        self, batch: Tensor, batch_idx: int
    ) -> Optional[STEP_OUTPUT]:
        (x, x_), y = batch
        y_hat = self._shared_step(x)
        _, y_hat_idx = torch.max(y_hat, dim=1)
        missclassified_idx = torch.where(y_hat_idx != y)[0]
        if x_[missclassified_idx].size(0):
            self.predict_step_outputs.append(x_[missclassified_idx])

    def on_predict_epoch_end(self, results: List[Any]) -> None:
        # Create grid of misclassified_idxs images
        X = torch.cat(self.predict_step_outputs, dim=0)
        misclassified_images = vutils.make_grid(
            X, nrow=int(math.sqrt(X.size(0))), padding=10,
        )
        self.logger.experiment.add_image(
            "Misclassified Images",
            misclassified_images,
            global_step=self.global_step,
        )
        self.predict_step_outputs.clear()

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[_TORCH_LRSCHEDULER]]:
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
