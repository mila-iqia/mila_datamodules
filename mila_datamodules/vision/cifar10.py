from pl_bolts.datamodules import CIFAR10DataModule as _CIFAR10DataModule

from mila_datamodules.clusters import adapt_dataset

from .vision_datamodule import VisionDataModule


class CIFAR10DataModule(_CIFAR10DataModule, VisionDataModule):
    dataset_cls = adapt_dataset(_CIFAR10DataModule.dataset_cls)
