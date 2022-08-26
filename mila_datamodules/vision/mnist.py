from pl_bolts.datamodules import MNISTDataModule as _MNISTDataModule

from mila_datamodules.clusters import adapt_dataset

from .vision_datamodule import VisionDataModule


class MNISTDataModule(_MNISTDataModule, VisionDataModule):
    dataset_cls = adapt_dataset(_MNISTDataModule.dataset_cls)
