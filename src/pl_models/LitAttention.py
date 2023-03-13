import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import AUROC, Accuracy, MetricCollection

from ..architecture import MILAttention


class LitAttention(pl.LightningModule):
    def __init__(self, lr: float = 0.001, **model_params):
        super().__init__()
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.train_metrics, self.val_metrics, self.test_metrics = [
            MetricCollection(
                [
                    Accuracy(task="multiclass", num_classes=2),
                    AUROC(task="multiclass", num_classes=2),
                ],
                prefix=prefix,
            )
            for prefix in ("train_", "val_", "test_")
        ]
        self.net = self._create_model(**model_params)
        self.validation_step_outputs = []
        self.example_input_array = torch.randn(32, 10, 784)

    def _create_model(self, **model_params):
        return MILAttention(**model_params)

    def forward(self, x):
        y_hat, atten_weights = self.net(x)
        return y_hat, atten_weights

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.forward(x)
        loss = self.criterion(y_hat, y)

        self.train_metrics(y_hat, y)
        self.log("train_loss", loss)
        self.log_dict(
            self.train_metrics,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, attn_w = self.forward(x)
        val_loss = self.criterion(y_hat, y)

        idx = torch.where(y == 1)[0]
        t = x[idx]
        attn_map = attn_w[0][idx, 0, 1:]
        self.validation_step_outputs.append([t, attn_map])

        self.val_metrics(y_hat, y)
        self.log("val_loss", val_loss)
        self.log_dict(
            self.val_metrics,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def on_validation_epoch_end(self):
        ts = []
        for t, attn_map in self.validation_step_outputs:
            B, N, C, H, W = t.size()
            t = t.permute(2, 0, 3, 1, 4).reshape(
                -1, B * H, N * W
            )  # (C, B, H, N, W) -> (C, B*H, N*W)
            if C == 1:
                t = t.repeat((3, 1, 1))
            attn_map = (
                attn_map.unsqueeze(1)
                .repeat((1, H, 1))
                .reshape(B * H, -1)  # (B, N) -> (B, H, N) -> (B*H, N)
            )
            attn_map = (
                attn_map.unsqueeze(2)
                .repeat((1, 1, W))
                .reshape(-1, N * W)  # (B*H, N) -> (B*H, N, W) -> (B*H, N*W)
            )
            t[0, :, :] = t[0, :, :] + attn_map.unsqueeze(
                0
            )  # (C, B*H, N*W) + (1, B*H, N*W) -> (C, B*H, N*W)
            ts.append(t.clamp(0, 1))
        tensorbord = self.logger.experiment
        tensorbord.add_image(
            "Attention Maps",
            torch.cat(ts, dim=1),
            self.global_step,
            dataformats="CHW",
        )
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.forward(x)
        test_loss = self.criterion(y_hat, y)

        self.test_metrics(y_hat, y)
        self.log("test_loss", test_loss)
        self.log_dict(
            self.test_metrics,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat, attn_w = self.forward(x)
        preds, _ = torch.max(y_hat, dim=1)
        return preds, attn_w

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    from pytorch_lightning.utilities.model_summary.model_summary import (
        ModelSummary,
    )

    net = MILAttention(input_dim=784, num_instances=10,)
    model = LitAttention(net=net)
    summary = ModelSummary(model)
    print(summary)
