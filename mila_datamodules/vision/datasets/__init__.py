"""Wrappers around the torchvision datasets, adapted to the current cluster.

These wrappers mostly just use a good default value for the 'root' argument based
on the current cluster.

They might also silently first copy the dataset to SLURM_TMPDIR before reading it, depending on the
size of the dataset.

NOTE: These wrappers don't do anything when outside a SLURM cluster.

TODO: For some of these datasets that have other required arguments, perhaps we could add "good"
default values for some of them? (Currently doing that in the VisionDataModule associated with
this dataset).
"""
from __future__ import annotations

import torchvision.datasets

from ._binary_mnist import BinaryEMNIST, BinaryMNIST
from ._mnist import MNIST
from .adapted_datasets import AdaptedDataset, adapt_dataset
from .prepare_dataset import prepare_dataset

# NOTE: These here use "Patched" versions of the datasets.
Caltech101 = adapt_dataset(torchvision.datasets.Caltech101)
Caltech256 = adapt_dataset(torchvision.datasets.Caltech256)
CelebA = adapt_dataset(torchvision.datasets.CelebA)
Cityscapes = adapt_dataset(torchvision.datasets.Cityscapes)
CIFAR10 = adapt_dataset(torchvision.datasets.CIFAR10)
CIFAR100 = adapt_dataset(torchvision.datasets.CIFAR100)
FashionMNIST = adapt_dataset(torchvision.datasets.FashionMNIST)
INaturalist = adapt_dataset(torchvision.datasets.INaturalist)
Places365 = adapt_dataset(torchvision.datasets.Places365)
STL10 = adapt_dataset(torchvision.datasets.STL10)
SVHN = adapt_dataset(torchvision.datasets.SVHN)

EMNIST = adapt_dataset(torchvision.datasets.EMNIST)
CocoDetection = adapt_dataset(torchvision.datasets.CocoDetection)
CocoCaptions = adapt_dataset(torchvision.datasets.CocoCaptions)


MNIST = adapt_dataset(MNIST)
BinaryMNIST = adapt_dataset(BinaryMNIST)
BinaryEMNIST = adapt_dataset(BinaryEMNIST)

""" TODO: (@lebrice): Chat with @abergeron about this:

1. `prepare_dataset` is a generic function with a "handler" for different types of datasets, e.g.
  - a general-purpose handler for ImageFolder datasets,
  - another more specific handler for ImageNet.

2. AdaptedDataset is a dummy base class that basically only modifies the `__init__` of the
dataset, by making its 'root' argument optional, and launches a "dataset preparation" routine
before calling the actual __init__ for the dataset, depending on the current cluster.

  - Currently, AdaptedDataset just calls `prepare_dataset`, using the un-initialized dataset object
    to dispatch based on the type.


Questions:
- Are there perhaps better / more "pythonic" ways of "patching" the datasets, while preserving 100%
the same API as the original dataset class?

Goals:
1. One-line change:

```python
from torchvision.datasets import ImageNet
from mila_datamodules.vision.datasets import ImageNet
```


- Do I actually need a "registry" listing all the stored datasets on all the clusters?
    - If not, then what are the alternatives? What do they look like?
    - If so, how much / what kind of information do I minimally require?
"""


# Import the `prepare_imagenet_dataset` function, so the handler gets registered for the generic
# `prepare_dataset` function that is called in `adapt_dataset`.
from .prepare_imagenet import prepare_imagenet_dataset  # noqa: E402

ImageNet = adapt_dataset(torchvision.datasets.ImageNet)
# todo: Add the other datasets here.
