# from pl_bolts.datamodules.binary_emnist_datamodule import BinaryEMNISTDataModule
from pl_bolts.datamodules.fashion_mnist_datamodule import (
    FashionMNISTDataModule as _FashionMNISTDataModule,
)

from mila_datamodules.vision.datasets import _adapt_dataset
from mila_datamodules.vision.vision_datamodule import VisionDataModule


class FashionMNISTDataModule(_FashionMNISTDataModule, VisionDataModule):
    dataset_cls = _adapt_dataset(_FashionMNISTDataModule.dataset_cls)
