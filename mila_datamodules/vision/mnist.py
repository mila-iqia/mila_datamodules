from pl_bolts.datamodules import MNISTDataModule as _MNISTDataModule

from .datasets import _adapt_dataset
from .vision_datamodule import VisionDataModule


class MNISTDataModule(_MNISTDataModule, VisionDataModule):
    dataset_cls = _adapt_dataset(_MNISTDataModule.dataset_cls)
