from typing import ClassVar

from .cifar10 import CIFAR10DataModule
from .vision_datamodule import VisionDataModule
from .vision_datamodule_test import VisionDataModuleTests


class TestCIFAR10DataModule(VisionDataModuleTests):
    DataModule: ClassVar[type[VisionDataModule]] = CIFAR10DataModule
