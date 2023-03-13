import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from ..utils import plot_to_image


class MissclassifiedImagesCallback(pl.Callback):
    def __init__(
        self,
        classes: List[str],
        dataloader_key: str = "val",
        epoch_interval: int = 1,
        num_images: int = 25,
        mean: Optional[Tuple[float]] = None,
        std: Optional[Tuple[float]] = None,
    ):
        super().__init__()
        self.classes = classes
        self.dataloader_key = dataloader_key
        self.epoch_interval = epoch_interval
        self.num_images = num_images
        self.mean = mean
        self.std = std
        self.validation_step_outputs: List[Dict[str, Tensor]] = []

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        pl_module.eval()
        with torch.no_grad():
            x, y = batch
            x = x.to(pl_module.device)
            y = y.to(pl_module.device)
            preds = pl_module(x).argmax(dim=1)
            miss_idx = torch.where(preds != y)[0]
            self.validation_step_outputs.append(
                {"x": x[miss_idx], "y": y[miss_idx], "preds": preds[miss_idx]}
            )
        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if trainer.current_epoch % self.epoch_interval:
            return

        X, Y, P = [], [], []
        for output in self.validation_step_outputs:
            X.append(output["x"])
            Y.append(output["y"])
            P.append(output["preds"])

        X = torch.cat(X, dim=0).cpu()
        Y = torch.cat(Y, dim=0).cpu()
        P = torch.cat(P, dim=0).cpu()

        # Convert normalized images to unnormalized images
        mean = (
            torch.tensor([0.5]).repeat(repeats=X.size(1))
            if self.mean is None
            else torch.tensor(self.mean)
        )
        std = (
            torch.tensor([0.5]).repeat(repeats=X.size(1))
            if self.std is None
            else torch.tensor(self.std)
        )
        X = X.permute(0, 2, 3, 1)
        X = X * std + mean
        X = X.clamp(0, 1)

        ### Visualize a 5x5 grid of the first 25 incorrect examples
        fig, axes = plt.subplots(
            5,
            5,
            figsize=(15, 15),
            subplot_kw={"xticks": [], "yticks": []},
            gridspec_kw=dict(hspace=0.3, wspace=0.01),
        )

        ### Print the index, true and predicted label and log to TensorBoard as text
        t_writer = trainer.logger.experiment
        for i, (ax, x, y, p) in enumerate(zip(axes.flatten(), X, Y, P)):
            ax.imshow(x, cmap="gray", vmin=0, vmax=1)
            ax.title.set_text(f"ID: {i}, True: {y}, Pred: {p}")
            t_writer.add_text(
                "Misclassified Images",
                f"True: {self.classes[y]}, Predicted: {self.classes[p]}",
                i,
            )

        ### Also log the figure itself as before
        t_writer.add_image("Misclassified Images", plot_to_image(fig))

    def _on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if trainer.current_epoch % self.epoch_interval:
            return

        # Get dataloader and model predictions
        dataloader = trainer.datamodule.val_dataloader()
        X = torch.empty(0, dtype=torch.long).to(pl_module.device)
        misclassified_idxs = torch.empty(0, dtype=torch.long).to(
            pl_module.device
        )
        with torch.no_grad():
            pl_module.eval()
            for x, y in dataloader:
                x = x.to(pl_module.device)
                y = y.to(pl_module.device)
                preds = pl_module(x).argmax(dim=1)
                miss_idx = torch.where(preds != y)[0]
                misclassified_idxs = torch.cat(
                    [misclassified_idxs, miss_idx], dim=0
                )
                X = torch.cat([X, x], dim=0)

        # Select random misclassified_idxs images
        if misclassified_idxs.size(0) > self.num_images:
            misclassified_idxs = misclassified_idxs.cpu().numpy()
            misclassified_idxs = np.random.choice(
                misclassified_idxs, size=self.num_images, replace=False
            )
            misclassified_idxs = torch.from_numpy(misclassified_idxs)
        X = X[misclassified_idxs]

        # Convert normalized images to unnormalized images
        mean = (
            torch.tensor([0.5]).repeat(repeats=X.size(1))
            if self.mean is None
            else torch.tensor(self.mean)
        )
        std = (
            torch.tensor([0.5]).repeat(repeats=X.size(1))
            if self.std is None
            else torch.tensor(self.std)
        )
        X = X.cpu().permute(0, 2, 3, 1)
        X = X * std + mean
        X = X.clamp(0, 1).permute(0, 3, 1, 2)

        # Create grid of misclassified_idxs images
        misclassified_images = vutils.make_grid(
            X, nrow=int(math.sqrt(self.num_images)), padding=10,
        )
        trainer.logger.experiment.add_image(
            "Misclassified Images",
            misclassified_images,
            global_step=trainer.global_step,
        )
        save_dir = os.path.join(trainer.logger.log_dir, "misclassified")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"epoch_{trainer.current_epoch}.png"
        )
        vutils.save_image(misclassified_images, save_path)
