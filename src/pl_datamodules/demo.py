import torch
from pytorch_lightning.demos.boring_classes import BoringDataModule


class FakeDataset1(BoringDataModule):
    def train_dataloader(self):
        print("⚡", "using FakeDataset1", "⚡")
        return torch.utils.data.DataLoader(self.random_train)


class FakeDataset2(BoringDataModule):
    def train_dataloader(self):
        print("⚡", "using FakeDataset2", "⚡")
        return torch.utils.data.DataLoader(self.random_train)
