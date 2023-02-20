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

from .adapted_datasets import AdaptedDataset, adapt_dataset, prepare_dataset
from .binary_mnist import BinaryEMNIST, BinaryMNIST
from .imagenet import ImageNet
from .mnist import MNIST

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


# todo: Add the other datasets here.
