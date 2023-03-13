from pytorch_lightning.demos.boring_classes import DemoModel
from torch import Tensor
from torchmetrics.classification.accuracy import Accuracy


class Model1(DemoModel):
    def configure_optimizers(self):
        print("⚡", "using Model1", "⚡")
        return super().configure_optimizers()


class Model2(DemoModel):
    def configure_optimizers(self):
        print("⚡", "using Model2", "⚡")
        return super().configure_optimizers()


class Model3(DemoModel):
    def __init__(
        self,
        embed_dim: int = 128,
        out_dim: int = 10,
        learning_rate: float = 0.02,
    ):
        super().__init__(out_dim, learning_rate)
        self.embed_dim = embed_dim

    def configure_optimizers(self):
        print("⚡", "using Model3", "⚡")
        return super().configure_optimizers()


class Model4(DemoModel):
    def __init__(
        self,
        embed_dim: int = 128,
        out_dim: int = 10,
        learning_rate: float = 0.02,
    ):
        super().__init__(out_dim, learning_rate)
        self.embed_dim = embed_dim
        self.val_acc = Accuracy("multiclass", num_classes=2)

    def validation_step(self, batch: Tensor, batch_idx: int):
        x = batch
        y_hat = self(x)
        loss = x.sum()
        self.val_acc(x, x)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc)
        return {"val_loss", loss}

    def configure_optimizers(self):
        print("⚡", "using Model4", "⚡")
        return super().configure_optimizers()
