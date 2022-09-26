"""Datamodule for CIFAR100."""
from mila_datamodules.vision.cifar10 import CIFAR10DataModule
from mila_datamodules.vision.datasets import CIFAR100
from mila_datamodules.vision.vision_datamodule import VisionDataModule


class CIFAR100DataModule(CIFAR10DataModule, VisionDataModule):
    name = "cifar100"
    dataset_cls = CIFAR100
    dims = (3, 32, 32)

    @property
    def num_samples(self) -> int:
        train_len, _ = self._get_splits(len_dataset=50_000)
        return train_len

    @property
    def num_classes(self) -> int:
        return 100
