import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTBagDataset(Dataset):
    def __init__(
        self, mnist_dataset: MNIST, bag_size: int = 10, target: int = 1
    ):
        self.bags = []
        self.bag_labels = []
        self.generate_bags(mnist_dataset, bag_size, target)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        bag = self.bags[index]
        label = self.bag_labels[index]
        return bag, label

    def generate_bags(
        self, mnist_dataset: MNIST, bag_size: int = 10, target: int = 1
    ):
        bag = []
        bag_label = 0
        for i, (img, label) in enumerate(mnist_dataset):
            bag.append(img)
            if label == target:
                bag_label = 1
            if len(bag) == bag_size:
                self.bags.append(torch.stack(bag, dim=0))
                self.bag_labels.append(bag_label)
                bag = []
                bag_label = 0


class MNISTBagDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_instances: int = 10,
        target_num: int = 1,
        data_dir: str = os.path.join(os.environ["root"], "data"),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.target_num = target_num
        self.data_dir = data_dir
        self.transform = transforms.ToTensor()

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
            mnist_train, mnist_val = random_split(mnist_full, [55000, 5000])
            self.mnist_train = MNISTBagDataset(
                mnist_dataset=mnist_train,
                bag_size=self.num_instances,
                target=self.target_num,
            )
            self.mnist_val = MNISTBagDataset(
                mnist_dataset=mnist_val,
                bag_size=self.num_instances,
                target=self.target_num,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNISTBagDataset(
                mnist_dataset=MNIST(
                    self.data_dir, train=False, transform=self.transform
                ),
                bag_size=self.num_instances,
                target=self.target_num,
            )

        if stage == "predict":
            self.mnist_predict = MNISTBagDataset(
                mnist_dataset=MNIST(
                    self.data_dir, train=False, transform=self.transform
                ),
                bag_size=self.num_instances,
                target=self.target_num,
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)
