"""Small patch for the MNIST dataset, to avoid re-downloading the data on the Beluga cluster."""
import os

from torchvision.datasets import MNIST as _MNIST

from mila_datamodules.clusters import CURRENT_CLUSTER
from mila_datamodules.clusters.cluster import Cluster

from .adapted_datasets import adapt_dataset

# TODO: The `dataset_files` entry for MNIST is actually wrong in the case of the Beluga cluster!
# TODO: Would it be better to just fix the naming issue / create symlinks on Beluga instead?


class MNIST(_MNIST):
    @property
    def folder_name(self):
        cls_name = self.__class__.__name__
        if CURRENT_CLUSTER is Cluster.Beluga:
            return cls_name.lower()
        return cls_name

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.folder_name, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.folder_name, "processed")


MNIST = adapt_dataset(MNIST)
