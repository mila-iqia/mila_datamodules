"""A registry of where each dataset is stored on each cluster."""
from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable

import pl_bolts.datasets
import torchvision.datasets as tvd

from mila_datamodules.clusters import CURRENT_CLUSTER
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.vision.datasets.binary_mnist import BinaryMNIST

dataset_files = {
    tvd.MNIST: ["MNIST"],
    tvd.CIFAR10: ["cifar-10-batches-py"],
    tvd.CIFAR100: ["cifar-100-python"],
    tvd.FashionMNIST: ["FashionMNIST"],
    tvd.Caltech101: ["caltech101"],
    tvd.Caltech256: ["caltech256"],
    tvd.CelebA: ["celeba"],
    tvd.Cityscapes: ["leftImg8bit", "gtFine", "gtCoarse"],
    tvd.INaturalist: ["2021_train", "2021_train_mini", "2021_valid"],
    tvd.Places365: [
        "categories_places365.txt",
        "places365_train_standard.txt",
        "data_large_standard",
        "places365_test.txt",
        "places365_val.txt",
        "val_large",
    ],
    tvd.STL10: ["stl10_binary"],
    tvd.SVHN: ["train_32x32.mat", "test_32x32.mat", "extra_32x32.mat"],
    # NOTE: We're not currently using the `test` folder for the coco datasets, since there isn't
    # an annotation file with it. (this makes sense, you probably have to submit predictions to the
    # website or something).
    tvd.CocoDetection: ["annotations", "test2017", "train2017", "val2017"],
    tvd.CocoCaptions: ["annotations", "test2017", "train2017", "val2017"],
    tvd.EMNIST: ["EMNIST"],
    # NOTE: This isn't quite true for the original class.
    # pl_bolts.datasets.BinaryMNIST: ["MNIST"],
    BinaryMNIST: ["MNIST"],
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


# TODO: Fill these in!

dataset_roots_per_cluster: dict[type, dict[Cluster, Path]] = defaultdict(dict)
""" The path to the `root` directory to use for each dataset type, for each cluster."""

# Add the known dataset locations on the mila cluster.
for dataset in [
    tvd.MNIST,
    tvd.CIFAR10,
    tvd.Caltech101,
    tvd.Caltech256,
    tvd.CelebA,
    tvd.Cityscapes,
    tvd.CocoCaptions,
    tvd.CocoDetection,
    tvd.CIFAR100,
    tvd.FashionMNIST,
    tvd.INaturalist,
    tvd.Places365,
    tvd.STL10,
    tvd.SVHN,
    # NOTE: BinaryMNIST from pl_bolts.datasets is buggy, and doesn't work out-of-the-box.
    # Our adapted version works though.
    # pl_bolts.datasets.BinaryMNIST,
    BinaryMNIST,
]:
    dataset_roots_per_cluster[dataset][Cluster.Mila] = Path("/network/datasets/torchvision")


# Add the known dataset locations on the mila cluster.
# TODO: On `Beluga`, the MNIST files are in a folder called 'mnist' instead of 'MNIST'.
for dataset in [
    tvd.MNIST,
    tvd.CIFAR10,
    tvd.CIFAR100,
    tvd.Cityscapes,
    tvd.CocoCaptions,
    tvd.CocoDetection,
    BinaryMNIST,
]:
    dataset_roots_per_cluster[dataset][Cluster.Beluga] = Path("/project/rpp-bengioy/data/curated")


too_large_for_slurm_tmpdir: set[Callable] = set()
""" Set of datasets which are too large to store in $SLURM_TMPDIR."""

# TODO: How about we adopt a de-centralized kind of registry, a bit like gym?
# In each dataset module, we could have a `mila_datamodules.register(name, locations={Mila: ...})`?


# TODO: Create a registry of the archives for each dataset, so that we can use these instead of
# copying the files individually.
dataset_archives_per_cluster: dict[type, dict[Cluster, list[str]]] = {
    tvd.Places365: {
        # TODO: Unclear if/how these archives should be used to construct the torchvision
        # Places365 dataset. (train_256(...).tar gets extracted to a `data_256` folder.. the
        # structure doesn't match what torchvision expects)
        # Cluster.Mila: [
        #     "/network/datasets/places365/256/train_256_places365standard.tar",
        #     "/network/datasets/places365/256/val_256.tar",
        #     "/network/datasets/places365/256/test_256.tar",
        # ]
    },
}


def is_stored_on_cluster(dataset_cls: type, cluster: Cluster | None = CURRENT_CLUSTER) -> bool:
    """Returns whether we know where to find the given dataset on the given cluster."""
    return (
        dataset_cls in dataset_roots_per_cluster
        and cluster in dataset_roots_per_cluster[dataset_cls]
    ) or (
        dataset_cls in dataset_archives_per_cluster
        and cluster in dataset_archives_per_cluster[dataset_cls]
    )


def get_dataset_root(
    dataset_cls: type,
    cluster: Cluster | None = None,
    default: str | None = None,
) -> str:
    """Gets the root directory to use to read the given dataset on the given cluster.

    If the dataset is not available on the cluster and `default` is set, then the default value is
    returned. Otherwise, if `default` is None, raises a NotImplementedError.
    """
    cluster = cluster or Cluster.current()

    github_issue_url = (
        f"https://github.com/lebrice/mila_datamodules/issues/new?"
        f"labels={cluster.name}&template=feature_request.md&"
        f"title=Feature%20request:%20{dataset_cls.__name__}%20on%20{cluster.name}"
    )

    if dataset_cls not in dataset_roots_per_cluster:
        for dataset in dataset_roots_per_cluster:
            # A class with the same name (e.g. our adapted datasets) was passed.
            if dataset.__name__ == dataset_cls.__name__:
                warnings.warn(
                    RuntimeWarning(f"Using dataset class {dataset} instead of {dataset_cls}")
                )
                dataset_cls = dataset
                break

    if not is_stored_on_cluster(dataset_cls, cluster):
        if default is not None:
            assert isinstance(default, str)
            return default

        # We don't know where this dataset is in this cluster.
        raise NotImplementedError(
            f"No known location for dataset {dataset_cls.__name__} on {cluster.name} "
            f"cluster!\n If you do know where it can be found on {cluster.name}, "
            f"please make an issue at {github_issue_url} so the registry can be updated."
        )

    if dataset_cls not in dataset_roots_per_cluster:
        # Unsupported dataset.
        raise NotImplementedError(
            f"No known location for dataset {dataset_cls.__name__} on any of the clusters!\n"
            f"If you do know where it can be found on {cluster.name}, or on any other "
            f"cluster, please make an issue at {github_issue_url} to add it to the registry."
        )
    dataset_root = dataset_roots_per_cluster[dataset_cls][cluster]
    return str(dataset_root)
