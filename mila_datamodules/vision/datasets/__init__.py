"""Wrappers around the torchvision datasets, adapted to the current cluster."""
from __future__ import annotations

import functools
from typing import Callable, TypeVar, cast

import pl_bolts.datasets
import torchvision.datasets
from torch.utils.data import Dataset

# from .utils import adapt_dataset
from typing_extensions import ParamSpec

from mila_datamodules.clusters import CURRENT_CLUSTER
from mila_datamodules.clusters.cluster import Cluster
from mila_datamodules.registry import (
    dataset_files,
    dataset_roots_per_cluster,
    get_dataset_root,
    too_large_for_slurm_tmpdir,
)
from mila_datamodules.utils import replace_arg_defaults
from mila_datamodules.vision.datasets.utils import adapted_constructor

T = TypeVar("T", bound=type)
D = TypeVar("D", bound=Dataset)
P = ParamSpec("P")
C = TypeVar("C", bound=Callable)


def _cache(fn: C) -> C:
    return functools.cache(fn)  # type: ignore


@_cache
def _adapt_dataset(dataset_type: type[D]) -> type[D]:
    """When not running on a cluster, returns the given input.

    When running on a cluster, returns a subclass of the given dataset that has a modified/adapted
    constructor.
    This constructor does a few things, but basically works like a three-level caching system.
    1. /network/datasets/torchvision (read-only "cache")
    2. $SCRATCH/cache/torch (writeable "cache")
    3. $SLURM_TMPDIR/data (writeable "cache", fastest storage).

    If the dataset isn't found on the cluster, it will be downloaded in $SCRATCH
    If the dataset fits in $SLURM_TMPDIR, it will be copied from wherever it is, and placed
    there.
    The dataset is then read from SLURM_TMPDIR.
    """
    if CURRENT_CLUSTER is None:
        return dataset_type  # Do nothing, since we're not on a SLURM cluster.

    dataset_subclass = type(
        dataset_type.__name__, (dataset_type,), {"__init__": adapted_constructor(dataset_type)}
    )
    dataset_subclass = cast("type[D]", dataset_subclass)
    return dataset_subclass


# NOTE: This won't do anything when not on a SLURM cluster.
MNIST = _adapt_dataset(torchvision.datasets.MNIST)
CIFAR10 = _adapt_dataset(torchvision.datasets.CIFAR10)
CIFAR100 = _adapt_dataset(torchvision.datasets.CIFAR100)
FashionMNIST = _adapt_dataset(torchvision.datasets.FashionMNIST)
Caltech101 = _adapt_dataset(torchvision.datasets.Caltech101)
Caltech256 = _adapt_dataset(torchvision.datasets.Caltech256)
CelebA = _adapt_dataset(torchvision.datasets.CelebA)
Cityscapes = _adapt_dataset(torchvision.datasets.Cityscapes)
INaturalist = _adapt_dataset(torchvision.datasets.INaturalist)
Places365 = _adapt_dataset(torchvision.datasets.Places365)
STL10 = _adapt_dataset(torchvision.datasets.STL10)
SVHN = _adapt_dataset(torchvision.datasets.SVHN)
CocoDetection = _adapt_dataset(torchvision.datasets.CocoDetection)
CocoCaptions = _adapt_dataset(torchvision.datasets.CocoCaptions)

EMNIST = _adapt_dataset(torchvision.datasets.EMNIST)
BinaryMNIST = _adapt_dataset(pl_bolts.datasets.BinaryMNIST)
BinaryEMNIST = _adapt_dataset(pl_bolts.datasets.BinaryEMNIST)
# todo: Add the other datasets here.
