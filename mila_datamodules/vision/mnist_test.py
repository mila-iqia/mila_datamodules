from typing import ClassVar

from .mnist import (
    BinaryEMNISTDataModule,
    BinaryMNISTDataModule,
    EMNISTDataModule,
    MNISTDataModule,
)
from .vision_datamodule import VisionDataModule
from .vision_datamodule_test import VisionDataModuleTests


class TestMNISTDataModule(VisionDataModuleTests):
    DataModule: ClassVar[type[VisionDataModule]] = MNISTDataModule


class TestEMNISTDataModule(VisionDataModuleTests):
    DataModule: ClassVar[type[VisionDataModule]] = EMNISTDataModule


class TestBinaryMNISTDataModule(VisionDataModuleTests):
    DataModule: ClassVar[type[VisionDataModule]] = BinaryMNISTDataModule


class TestBinaryEMNISTDataModule(VisionDataModuleTests):
    DataModule: ClassVar[type[VisionDataModule]] = BinaryEMNISTDataModule
