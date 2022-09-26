from typing import ClassVar

from .cifar100 import CIFAR100DataModule
from .vision_datamodule import VisionDataModule
from .vision_datamodule_test import VisionDataModuleTests


class TestCIFAR100DataModule(VisionDataModuleTests):
    DataModule: ClassVar[type[VisionDataModule]] = CIFAR100DataModule
