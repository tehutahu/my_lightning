import os
from typing import Any, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTPredict(MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)
        raw = self.data[index].unsqueeze(0)
        return (img, raw), target

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__base__.__name__, "raw")


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 2,
        data_dir: str = os.path.join(os.environ["root"], "data"),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000]
            )

        # Assign validate dataset for use in dataloader(s)
        if stage == "validate":
            mnist_full = MNIST(
                self.data_dir, train=True, transform=self.transform
            )
            _, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.mnist_predict = MNISTPredict(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
