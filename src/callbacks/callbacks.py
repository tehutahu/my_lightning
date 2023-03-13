import math
import os
from typing import Any, Dict, List, Optional, Tuple

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
        epoch_interval: int = 1,
        num_images: int = 25,
        mean: Optional[Tuple[float]] = None,
        std: Optional[Tuple[float]] = None,
    ):
        super().__init__()
        self.classes = classes
        self.epoch_interval = epoch_interval
        self.num_images = num_images
        self.mean = mean
        self.std = std
        self.validation_step_outputs: List[Dict[str, Tensor]] = []

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        print(f"⚡⚡⚡ {self.__class__.__name__} ⚡⚡⚡")

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
                {"x": x[miss_idx], "y": y[miss_idx], "p": preds[miss_idx]}
            )
        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if trainer.current_epoch % self.epoch_interval:
            self.validation_step_outputs.clear()
            return

        X, Y, P = [], [], []
        for output in self.validation_step_outputs:
            X.append(output["x"])
            Y.append(output["y"])
            P.append(output["p"])

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

        # Visualize a grid of the first "num_images" incorrect examples
        N = self.num_images if self.num_images < X.size(0) else X.size(0)
        n_row = int(math.sqrt(N))
        n_col = math.ceil(N // n_row)
        fig, axes = plt.subplots(
            n_row,
            n_col,
            figsize=(15, 15),
            subplot_kw={"xticks": [], "yticks": []},
            gridspec_kw=dict(hspace=0.3, wspace=0.01),
        )

        # Print the index, true and predicted label and log to TensorBoard as text
        t_writer = trainer.logger.experiment
        for ax, x, y, p in zip(axes.flatten(), X, Y, P):
            ax.imshow(x, cmap="gray", vmin=0, vmax=1)
            ax.title.set_text(f"Y: {y}, Y_hat: {p}")
            # t_writer.add_text(
            #     "Misclassified Images",
            #     f"True: {self.classes[y]}, Predicted: {self.classes[p]}",
            #     i,
            # )

        # Also log the figure itself as before
        misclassified_images = plot_to_image(fig)
        t_writer.add_image(
            "Misclassified Images",
            plot_to_image(fig),
            global_step=trainer.global_step,
        )
        save_dir = os.path.join(trainer.logger.log_dir, "misclassified")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"epoch_{trainer.current_epoch}.png"
        )
        vutils.save_image(misclassified_images, save_path)

        self.validation_step_outputs.clear()
