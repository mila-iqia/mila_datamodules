from typing import ClassVar

from .fashion_mnist import FashionMNISTDataModule
from .vision_datamodule import VisionDataModule
from .vision_datamodule_test import VisionDataModuleTests


class TestFashionMNISTDataModule(VisionDataModuleTests):
    DataModule: ClassVar[type[VisionDataModule]] = FashionMNISTDataModule
