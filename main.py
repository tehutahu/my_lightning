# main.py
import os

os.environ.update(root=os.path.abspath(os.path.dirname(__file__)))

from pytorch_lightning.cli import LightningCLI
from src.pl_datamodules import (
    FakeDataset1,
    MNISTBagDataModule,
    MNISTDataModule,
)
from src.pl_models import MLP, LitAttention, Model1


def cli_main():
    cli = LightningCLI()
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
