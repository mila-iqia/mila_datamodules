from pl_bolts.datamodules import CIFAR10DataModule as _CIFAR10DataModule

from mila_datamodules.vision.datasets import CIFAR10

from .vision_datamodule import VisionDataModule


class CIFAR10DataModule(_CIFAR10DataModule, VisionDataModule):
    dataset_cls = CIFAR10
