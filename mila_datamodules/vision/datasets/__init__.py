"""Wrappers around the torchvision datasets, adapted to the current cluster."""
from __future__ import annotations

import functools
from typing import Callable, TypeVar

import torchvision.datasets
from torch.utils.data import Dataset

# from .utils import adapt_dataset
from typing_extensions import ParamSpec

from mila_datamodules.vision.datasets.utils import adapted_constructor

D = TypeVar("D", bound=Dataset)
P = ParamSpec("P")
C = TypeVar("C", bound=Callable)


def _cache(fn: C) -> C:
    return functools.cache(fn)  # type: ignore


@_cache
def _adapt_dataset(dataset_type: type[D]) -> type[D]:
    t: type = type(
        dataset_type.__name__, (dataset_type,), {"__init__": adapted_constructor(dataset_type)}
    )
    assert issubclass(t, dataset_type)
    return t  # type: ignore


MNIST = _adapt_dataset(torchvision.datasets.MNIST)
FashionMNIST = _adapt_dataset(torchvision.datasets.FashionMNIST)
CIFAR10 = _adapt_dataset(torchvision.datasets.CIFAR10)
CIFAR100 = _adapt_dataset(torchvision.datasets.CIFAR100)
# todo: Add the other datasets here.
