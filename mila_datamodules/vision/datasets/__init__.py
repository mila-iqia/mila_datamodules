"""Wrappers around the torchvision datasets, adapted to the current cluster."""
from __future__ import annotations

import functools
from typing import Callable, TypeVar, cast

import cv2
import torchvision.datasets
from torch.utils.data import Dataset

# from .utils import adapt_dataset
from typing_extensions import ParamSpec

from mila_datamodules.clusters import CURRENT_CLUSTER
from mila_datamodules.clusters.cluster_enum import ClusterType
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
if CURRENT_CLUSTER is ClusterType.MILA:
    # TODO: Unsure if we want to go there, but it might be useful to include good default values
    # for these kinds of datasets which require some argument that is a path to some annotation
    # file.
    # NOTE: In this case here, the dataset can now be created without any arguments, but if a value
    # is passed for the `annFile` argument, it will be used.
    CocoDetection = replace_arg_defaults(
        CocoDetection, annFile="/network/datasets/torchvision/annotations/captions_train2017.json"
    )

# todo: Add the other datasets here.
