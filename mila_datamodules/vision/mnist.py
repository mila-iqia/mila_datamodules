from pl_bolts.datamodules import BinaryEMNISTDataModule as _BinaryEMNISTDataModule
from pl_bolts.datamodules import BinaryMNISTDataModule as _BinaryMNISTDataModule
from pl_bolts.datamodules import EMNISTDataModule as _EMNISTDataModule
from pl_bolts.datamodules import MNISTDataModule as _MNISTDataModule

from mila_datamodules.vision.datasets import EMNIST, MNIST, BinaryEMNIST, BinaryMNIST

from .vision_datamodule import VisionDataModule


class MNISTDataModule(_MNISTDataModule, VisionDataModule):
    dataset_cls = MNIST


class BinaryMNISTDataModule(_BinaryMNISTDataModule, VisionDataModule):
    dataset_cls = BinaryMNIST


# NOTE: These two aren't stored on the Mila cluster atm. Therefore, they should be downloaded to
# $SCRATCH/data the first time they are used.


class EMNISTDataModule(_EMNISTDataModule, VisionDataModule):
    dataset_cls = EMNIST


class BinaryEMNISTDataModule(_BinaryEMNISTDataModule, VisionDataModule):
    dataset_cls = BinaryEMNIST
