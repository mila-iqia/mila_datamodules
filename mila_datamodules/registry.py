"""A registry of where each dataset is stored on each cluster."""
from __future__ import annotations

import warnings
from typing import Callable

import pl_bolts.datasets
import torchvision.datasets as tvd

from mila_datamodules.clusters import CURRENT_CLUSTER
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.clusters.utils import get_scratch_dir
from mila_datamodules.utils import all_files_exist
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

_mila_torchvision_dir = "/network/datasets/torchvision"
_beluga_curated_datasets_dir = "/project/rpp-bengioy/data/curated"

# TODO: Fill this in!
# TODO: Also allow configuring / updating these using a config file instead of having to hard-code
# things.
# Add the known dataset locations on the mila cluster.

dataset_roots_per_cluster = {
    Cluster.Mila: {
        tvd.MNIST: _mila_torchvision_dir,
        tvd.MNIST: _mila_torchvision_dir,
        tvd.CIFAR10: _mila_torchvision_dir,
        tvd.Caltech101: _mila_torchvision_dir,
        tvd.Caltech256: _mila_torchvision_dir,
        tvd.CelebA: _mila_torchvision_dir,
        tvd.Cityscapes: _mila_torchvision_dir,
        tvd.CocoCaptions: _mila_torchvision_dir,
        tvd.CocoDetection: _mila_torchvision_dir,
        tvd.CIFAR100: _mila_torchvision_dir,
        tvd.FashionMNIST: _mila_torchvision_dir,
        tvd.INaturalist: _mila_torchvision_dir,
        tvd.Places365: _mila_torchvision_dir,
        tvd.STL10: _mila_torchvision_dir,
        tvd.SVHN: _mila_torchvision_dir,
        # NOTE: BinaryMNIST from pl_bolts.datasets is buggy, and doesn't work out-of-the-box.
        # Our adapted version works though.
        # pl_bolts.datasets.BinaryMNIST,
        BinaryMNIST: _mila_torchvision_dir,
    },
    Cluster.Beluga: {
        tvd.MNIST: "/project/rrp-bengioy/data",
        tvd.CIFAR10: _beluga_curated_datasets_dir,
        tvd.Cityscapes: _beluga_curated_datasets_dir,
        tvd.CocoCaptions: _beluga_curated_datasets_dir,
        tvd.CocoDetection: _beluga_curated_datasets_dir,
        BinaryMNIST: "/project/rrp-bengioy/data",
    },
}
"""For each cluster, for each type of dataset, the value of `root` to use to load the dataset."""


# TODO: Create a registry of the archives for each dataset, so that we can use these instead of
# copying the files individually.

dataset_archives_per_cluster: dict[Cluster, dict[type, list[str]]] = {
    Cluster.Mila: {
        # TODO: Unclear if/how these archives should be used to construct the torchvision
        # Places365 dataset. (train_256(...).tar gets extracted to a `data_256` folder.. the
        # structure doesn't match what torchvision expects)
        tvd.Places365: [
            "/network/datasets/places365/256/train_256_places365standard.tar",
            "/network/datasets/places365/256/val_256.tar",
            "/network/datasets/places365/256/test_256.tar",
        ]
    },
    Cluster.Beluga: {
        tvd.CIFAR100: ["/project/rpp-bengioy/data/curated/cifar100/cifar-100-python.tar.gz"]
    },
}
"""For each cluster, for each type of dataset, the list of archives needed to load the dataset."""


too_large_for_slurm_tmpdir: set[Callable] = set()
"""Set of datasets which are too large to store in $SLURM_TMPDIR."""

# TODO: How about we adopt a de-centralized kind of registry, a bit like gym?
# In each dataset module, we could have a `mila_datamodules.register(name, locations={Mila: ...})`?


# NOTE: Does it make sense to allow passing `cluster=None` here? (in the sense that we're not on a cluster?)


def is_stored_on_cluster(dataset_cls: type, cluster: Cluster | None = CURRENT_CLUSTER) -> bool:
    """Returns whether we know where to find the given dataset on the given cluster."""
    if cluster in dataset_roots_per_cluster and dataset_cls in dataset_roots_per_cluster[cluster]:
        return True
    if (
        cluster in dataset_archives_per_cluster
        and dataset_cls in dataset_archives_per_cluster[cluster]
    ):
        return True
    # TODO: This check here won't work when passing a subclass of the dataset. Might want to
    # make this check into a function like `has_known_required_files` ? (something clearer and
    # simpler) than that.

    if dataset_cls in dataset_files:
        # We know which files are needed for that dataset!
        # Dynamically check if we have all the files for that dataset (where in the cluster?)
        # (in the SCRATCH directory for now).
        files_required_to_load_dataset = dataset_files[dataset_cls]
        return all_files_exist(
            required_files=files_required_to_load_dataset,
            base_dir=get_scratch_dir(),
        )

    if cluster in dataset_archives_per_cluster:
        if dataset_cls in dataset_archives_per_cluster[cluster]:
            # We know where to find the archives for that dataset on that cluster!
            return True

    # We don't know where to find the dataset files on that cluster.
    return False


def locate_dataset_root_on_cluster(
    dataset_cls: type,
    cluster: Cluster | None = CURRENT_CLUSTER,
    default: str | None = None,
) -> str:
    """Gets the root directory to use to read the given dataset on the given cluster.

    If the dataset is not available on the cluster and `default` is set, then the default value is
    returned. Otherwise, if `default` is None, raises a NotImplementedError.
    """
    # TODO: Unclear what to do when not on a cluster. Should this just not be used?
    if cluster is None:
        if default is not None:
            return default
        raise RuntimeError(
            f"Was asked what root directory is appropriate for dataset type {dataset_cls} while "
            f"not on a SLURM cluster, and no default value was passed. "
            # NOTE: Now raising an error instead, just to avoid any surprises.
            # f"Returning the current directory. "
        )
        # return str(Path.cwd())

    cluster_name = cluster.name if cluster is not None else "local"
    github_issue_url = (
        f"https://github.com/lebrice/mila_datamodules/issues/new?"
        f"labels={cluster_name}&template=feature_request.md&"
        f"title=Feature%20request:%20{dataset_cls.__name__}%20on%20{cluster_name}"
    )

    if (
        cluster in dataset_roots_per_cluster
        and dataset_cls not in dataset_roots_per_cluster[cluster]
    ):
        for dataset in dataset_roots_per_cluster[cluster]:
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
            f"No known location for dataset {dataset_cls.__name__} on {cluster_name} "
            f"cluster!\n If you do know where it can be found on {cluster_name}, "
            f"please make an issue at {github_issue_url} so the registry can be updated."
        )

    if cluster not in dataset_roots_per_cluster:
        raise NotImplementedError(
            f"Don't know where datasets are stored in cluster {cluster_name}! \n"
            f"If you do know where it can be found on {cluster_name}, or on any other "
            f"cluster, 🙏 please make an issue at {github_issue_url} to add it to the registry 🙏."
        )

    dataset_root = dataset_roots_per_cluster[cluster].get(dataset_cls)
    if dataset_cls is None:
        # Unsupported dataset?
        raise NotImplementedError(
            f"No known location for dataset {dataset_cls.__name__} on any of the clusters!\n"
            f"If you do know where it can be found on {cluster_name}, or on any other "
            f"cluster, please make an issue at {github_issue_url} to add it to the registry."
        )
    return str(dataset_root)
