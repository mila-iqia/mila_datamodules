"""A registry of where each dataset is stored on each cluster."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pl_bolts.datasets
import torchvision.datasets as tvd

from mila_datamodules.clusters.cluster import Cluster

dataset_files = {
    tvd.MNIST: ["MNIST"],
    tvd.CIFAR10: ["cifar-10-batches-py"],
    tvd.CIFAR100: ["cifar-100-python"],
    tvd.FashionMNIST: ["FashionMNIST"],
    tvd.Caltech101: ["caltech101"],
    tvd.Caltech256: ["caltech256"],
    tvd.CelebA: ["celeba"],
    tvd.Cityscapes: ["cityscapes"],
    tvd.INaturalist: ["inat"],
    tvd.Places365: ["places365"],
    tvd.STL10: ["stl10"],
    tvd.SVHN: ["SVHN"],
    tvd.CocoDetection: ["annotations", "test2017", "train2017", "val2017"],
    tvd.CocoCaptions: ["annotations", "test2017", "train2017", "val2017"],
    tvd.EMNIST: ["EMNIST"],
    pl_bolts.datasets.BinaryMNIST: ["MNIST"],
    pl_bolts.datasets.BinaryEMNIST: ["EMNIST"],
}
"""A map of the folder/files associated with each dataset type, relative to the `root_dir`. This is
roughly the list of files/folders that would be downloaded when creating the dataset with
`download=True`.

NOTE: An entry being in this dict *does not* mean that this dataset is available on the cluster!
This is simply a list of the files that are expected to be present in the `root` directory of each
dataset type in order for it to work.

These files are copied over to $SCRATCH or $SLURM_TMPDIR before the dataset is read, depending on
the type of dataset.
"""

dataset_roots_per_cluster: dict[Cluster, dict[type, Path]] = {
    Cluster.MILA: {
        k: Path("/network/datasets/torchvision")
        for k in [
            tvd.MNIST,
            tvd.CIFAR10,
            tvd.CIFAR100,
            tvd.FashionMNIST,
            tvd.Caltech101,
            tvd.Caltech256,
            tvd.CelebA,
            tvd.Cityscapes,
            tvd.INaturalist,
            tvd.Places365,
            tvd.STL10,
            tvd.SVHN,
            tvd.CocoDetection,
            pl_bolts.datasets.BinaryMNIST,
        ]
    },
    # TODO: Fill these in!
    Cluster.BELUGA: {},
    Cluster.GRAHAM: {},
    Cluster.CEDAR: {},
}
""" The path to the `root` path to use to read each dataset type, for each cluster.

TODO: This unified torchvision_root structure might not be the same for the CC clusters.
"""


too_large_for_slurm_tmpdir: set[Callable] = set()
""" Set of datasets which are too large to store in $SLURM_TMPDIR."""


def get_dataset_root(dataset_cls: type, cluster: Cluster | None = None) -> str:
    """Gets the root directory to use to read the given dataset on the given cluster.

    Raises a NotImplementedError if the dataset location isn't known for that cluster.
    """
    cluster = cluster or Cluster.current()
    dataset_roots = dataset_roots_per_cluster[cluster]
    if dataset_cls not in dataset_roots:
        raise NotImplementedError(
            f"Don't know yet where dataset `{dataset_cls}` is stored on cluster {cluster}!"
        )
    return str(dataset_roots[dataset_cls])
