from pl_bolts.datamodules.fashion_mnist_datamodule import (
    FashionMNISTDataModule as _FashionMNISTDataModule,
)

from mila_datamodules.vision.datasets import FashionMNIST
from mila_datamodules.vision.vision_datamodule import VisionDataModule


class FashionMNISTDataModule(_FashionMNISTDataModule, VisionDataModule):
    dataset_cls = FashionMNIST
