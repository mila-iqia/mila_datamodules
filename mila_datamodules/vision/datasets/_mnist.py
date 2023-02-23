"""Small patch for the MNIST dataset, to avoid re-downloading the data on the Beluga cluster."""
import os

from torchvision.datasets import MNIST as _MNIST

from mila_datamodules.clusters import CURRENT_CLUSTER
from mila_datamodules.clusters.cluster import Cluster


class MNIST(_MNIST):
    # FIXME: Unfortunate that we need to do this..
    # Fixes a mismatch in the dataset file names for the Beluga cluster.
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
